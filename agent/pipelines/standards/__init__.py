"""
Standards Lookup Pipeline - Fetches and returns relevant regulations

This module provides functionality to fetch, categorize, and return relevant
regulatory standards and compliance requirements from various sources.
"""

import asyncio
import json

# Optional async imports - only needed for actual API calls
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import re
from pathlib import Path
from pydantic import BaseModel, Field
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StandardType(Enum):
    """Enumeration for different types of standards"""
    REGULATION = "regulation"
    GUIDELINE = "guideline"
    POLICY = "policy"
    FRAMEWORK = "framework"
    CERTIFICATION = "certification"
    BEST_PRACTICE = "best_practice"


class ComplianceLevel(Enum):
    """Enumeration for compliance levels"""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    DEPRECATED = "deprecated"


class Jurisdiction(Enum):
    """Enumeration for different jurisdictions"""
    US_FEDERAL = "us_federal"
    US_STATE = "us_state"
    EU = "eu"
    UK = "uk"
    CANADA = "canada"
    AUSTRALIA = "australia"
    INTERNATIONAL = "international"


class StandardsOrganization(Enum):
    """Enumeration for different standards organizations"""
    OSHA = "osha"
    NIOSH = "niosh"
    NFPA = "nfpa"
    ANSI = "ansi"
    ISO = "iso"
    IEC = "iec"
    ASTM = "astm"
    ASME = "asme"
    IEEE = "ieee"
    OTHER = "other"


@dataclass
class Standard:
    """Represents a regulatory standard or guideline"""
    id: str
    title: str
    description: str
    standard_type: StandardType
    compliance_level: ComplianceLevel
    jurisdiction: Jurisdiction
    version: str
    effective_date: datetime
    organization: StandardsOrganization = StandardsOrganization.OTHER
    standard_number: Optional[str] = None  # e.g., "1910.178", "NFPA 70E"
    expiration_date: Optional[datetime] = None
    source_url: Optional[str] = None
    keywords: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    related_standards: Set[str] = field(default_factory=set)
    requirements: List[str] = field(default_factory=list)
    penalties: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StandardsLookupPipeline:
    """
    Pipeline for fetching and managing regulatory standards
    
    Features:
    - Multi-source standards fetching
    - Real-time regulatory updates
    - Compliance checking and validation
    - Standards categorization and search
    - Change detection and notifications
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the standards lookup pipeline"""
        self.config = self._load_config(config_path)
        self.standards_cache: Dict[str, Standard] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.update_interval = timedelta(hours=self.config.get("update_interval_hours", 24))
        self.last_update = datetime.now() - self.update_interval
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {
            "sources": {
                "federal_register": {
                    "enabled": True,
                    "api_url": "https://api.federalregister.gov/v1/documents.json",
                    "rate_limit": 1000  # requests per hour
                },
                "eu_legislation": {
                    "enabled": True,
                    "api_url": "https://eur-lex.europa.eu/api/",
                    "rate_limit": 500
                },
                "local_cache": {
                    "enabled": True,
                    "cache_path": "data/standards_cache.json"
                }
            },
            "update_interval_hours": 24,
            "max_concurrent_requests": 10,
            "timeout_seconds": 30,
            "retry_attempts": 3
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        if HAS_AIOHTTP:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config["timeout_seconds"]),
                connector=aiohttp.TCPConnector(limit=self.config["max_concurrent_requests"])
            )
        else:
            self.session = None
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_standards(self, 
                            keywords: Optional[List[str]] = None,
                            jurisdiction: Optional[Jurisdiction] = None,
                            standard_type: Optional[StandardType] = None,
                            force_update: bool = False) -> List[Standard]:
        """
        Fetch standards from configured sources
        
        Args:
            keywords: Optional keywords to filter by
            jurisdiction: Optional jurisdiction filter
            standard_type: Optional standard type filter
            force_update: Force update even if cache is fresh
            
        Returns:
            List[Standard]: List of fetched standards
        """
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not available. Using cached standards only.")
            return self._filter_standards(keywords, jurisdiction, standard_type)
        
        if not self.session:
            raise RuntimeError("Pipeline not initialized. Use async context manager.")
        
        # Check if update is needed
        if not force_update and datetime.now() - self.last_update < self.update_interval:
            logger.info("Using cached standards data")
            return self._filter_standards(keywords, jurisdiction, standard_type)
        
        logger.info("Fetching fresh standards data")
        fetched_standards = []
        
        # Fetch from enabled sources
        tasks = []
        for source_name, source_config in self.config["sources"].items():
            if source_config.get("enabled", False) and source_name != "local_cache":
                task = self._fetch_from_source(source_name, source_config)
                tasks.append(task)
        
        # Execute all fetch tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error fetching standards: {result}")
                else:
                    fetched_standards.extend(result)
        
        # Update cache
        self._update_cache(fetched_standards)
        self.last_update = datetime.now()
        
        return self._filter_standards(keywords, jurisdiction, standard_type)
    
    async def _fetch_from_source(self, source_name: str, source_config: Dict) -> List[Standard]:
        """Fetch standards from a specific source"""
        try:
            if source_name == "federal_register":
                return await self._fetch_federal_register(source_config)
            elif source_name == "eu_legislation":
                return await self._fetch_eu_legislation(source_config)
            else:
                logger.warning(f"Unknown source: {source_name}")
                return []
        except Exception as e:
            logger.error(f"Error fetching from {source_name}: {e}")
            return []
    
    async def _fetch_federal_register(self, config: Dict) -> List[Standard]:
        """Fetch standards from US Federal Register"""
        standards = []
        
        try:
            # Search for recent regulations
            params = {
                "per_page": 100,
                "order": "newest",
                "publication_date": {
                    "gte": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                }
            }
            
            async with self.session.get(config["api_url"], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for doc in data.get("results", []):
                        standard = self._parse_federal_register_document(doc)
                        if standard:
                            standards.append(standard)
                else:
                    logger.error(f"Federal Register API error: {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching Federal Register data: {e}")
        
        return standards
    
    async def _fetch_eu_legislation(self, config: Dict) -> List[Standard]:
        """Fetch standards from EU legislation database"""
        standards = []
        
        try:
            # This is a simplified implementation
            # In practice, you'd need to implement the specific EU API calls
            logger.info("EU legislation fetching not fully implemented")
            
        except Exception as e:
            logger.error(f"Error fetching EU legislation: {e}")
        
        return standards
    
    def _parse_federal_register_document(self, doc: Dict) -> Optional[Standard]:
        """Parse a Federal Register document into a Standard object"""
        try:
            # Extract relevant information
            title = doc.get("title", "")
            summary = doc.get("summary", "")
            publication_date = datetime.strptime(doc.get("publication_date", ""), "%Y-%m-%d")
            
            # Determine standard type based on document type
            doc_type = doc.get("type", "").lower()
            if "regulation" in doc_type:
                standard_type = StandardType.REGULATION
            elif "guidance" in doc_type:
                standard_type = StandardType.GUIDELINE
            else:
                standard_type = StandardType.POLICY
            
            # Extract keywords from title and summary
            keywords = self._extract_keywords(title + " " + summary)
            
            # Create standard ID
            standard_id = hashlib.md5(f"{title}_{publication_date}".encode()).hexdigest()[:16]
            
            standard = Standard(
                id=standard_id,
                title=title,
                description=summary,
                standard_type=standard_type,
                compliance_level=ComplianceLevel.MANDATORY,  # Default for regulations
                jurisdiction=Jurisdiction.US_FEDERAL,
                version="1.0",
                effective_date=publication_date,
                source_url=doc.get("html_url"),
                keywords=keywords,
                categories=self._categorize_standard(keywords),
                last_updated=datetime.now(),
                metadata={"federal_register_id": doc.get("document_number")}
            )
            
            return standard
            
        except Exception as e:
            logger.error(f"Error parsing Federal Register document: {e}")
            return None
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text"""
        # Simple keyword extraction - in practice, you'd use NLP libraries
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = {word for word in words if word not in stop_words and len(word) > 3}
        
        return keywords
    
    def _categorize_standard(self, keywords: Set[str]) -> Set[str]:
        """Categorize standard based on keywords"""
        categories = set()
        
        # Define category mappings
        category_mappings = {
            "privacy": {"privacy", "data", "personal", "gdpr", "ccpa"},
            "security": {"security", "cyber", "encryption", "authentication"},
            "financial": {"financial", "banking", "payment", "fintech"},
            "healthcare": {"health", "medical", "hipaa", "patient"},
            "environmental": {"environment", "climate", "carbon", "sustainability"},
            "labor": {"employment", "labor", "workplace", "safety"},
            "technology": {"technology", "software", "ai", "algorithm"}
        }
        
        for category, category_keywords in category_mappings.items():
            if keywords.intersection(category_keywords):
                categories.add(category)
        
        return categories
    
    def _update_cache(self, standards: List[Standard]) -> None:
        """Update the standards cache"""
        for standard in standards:
            self.standards_cache[standard.id] = standard
        
        # Save to local cache if enabled
        cache_config = self.config["sources"].get("local_cache", {})
        if cache_config.get("enabled", False):
            self._save_cache_to_file(cache_config.get("cache_path", "data/standards_cache.json"))
    
    def _save_cache_to_file(self, cache_path: str) -> None:
        """Save cache to file"""
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {}
            for standard_id, standard in self.standards_cache.items():
                cache_data[standard_id] = {
                    "id": standard.id,
                    "title": standard.title,
                    "description": standard.description,
                    "standard_type": standard.standard_type.value,
                    "compliance_level": standard.compliance_level.value,
                    "jurisdiction": standard.jurisdiction.value,
                    "version": standard.version,
                    "effective_date": standard.effective_date.isoformat(),
                    "expiration_date": standard.expiration_date.isoformat() if standard.expiration_date else None,
                    "source_url": standard.source_url,
                    "keywords": list(standard.keywords),
                    "categories": list(standard.categories),
                    "related_standards": list(standard.related_standards),
                    "requirements": standard.requirements,
                    "penalties": standard.penalties,
                    "last_updated": standard.last_updated.isoformat(),
                    "metadata": standard.metadata
                }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving cache to file: {e}")
    
    def _load_cache_from_file(self, cache_path: str) -> None:
        """Load cache from file"""
        try:
            if Path(cache_path).exists():
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                
                for standard_id, data in cache_data.items():
                    standard = Standard(
                        id=data["id"],
                        title=data["title"],
                        description=data["description"],
                        standard_type=StandardType(data["standard_type"]),
                        compliance_level=ComplianceLevel(data["compliance_level"]),
                        jurisdiction=Jurisdiction(data["jurisdiction"]),
                        version=data["version"],
                        effective_date=datetime.fromisoformat(data["effective_date"]),
                        expiration_date=datetime.fromisoformat(data["expiration_date"]) if data["expiration_date"] else None,
                        source_url=data["source_url"],
                        keywords=set(data["keywords"]),
                        categories=set(data["categories"]),
                        related_standards=set(data["related_standards"]),
                        requirements=data["requirements"],
                        penalties=data["penalties"],
                        last_updated=datetime.fromisoformat(data["last_updated"]),
                        metadata=data["metadata"]
                    )
                    self.standards_cache[standard_id] = standard
                    
        except Exception as e:
            logger.error(f"Error loading cache from file: {e}")
    
    def _filter_standards(self, 
                         keywords: Optional[List[str]] = None,
                         jurisdiction: Optional[Jurisdiction] = None,
                         standard_type: Optional[StandardType] = None) -> List[Standard]:
        """Filter standards based on criteria"""
        filtered = list(self.standards_cache.values())
        
        if keywords:
            keyword_set = set(keyword.lower() for keyword in keywords)
            filtered = [
                s for s in filtered 
                if keyword_set.intersection(s.keywords) or 
                   any(keyword.lower() in s.title.lower() or keyword.lower() in s.description.lower() 
                       for keyword in keywords)
            ]
        
        if jurisdiction:
            filtered = [s for s in filtered if s.jurisdiction == jurisdiction]
        
        if standard_type:
            filtered = [s for s in filtered if s.standard_type == standard_type]
        
        return filtered
    
    def search_standards(self, query: str, limit: int = 10) -> List[Standard]:
        """
        Search standards by query string
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List[Standard]: Matching standards
        """
        query_lower = query.lower()
        scored_standards = []
        
        for standard in self.standards_cache.values():
            score = 0
            
            # Title match (highest weight)
            if query_lower in standard.title.lower():
                score += 10
            
            # Description match
            if query_lower in standard.description.lower():
                score += 5
            
            # Keyword match
            query_words = set(query_lower.split())
            keyword_matches = len(query_words.intersection(standard.keywords))
            score += keyword_matches * 3
            
            # Category match
            category_matches = len(query_words.intersection(standard.categories))
            score += category_matches * 2
            
            if score > 0:
                scored_standards.append((score, standard))
        
        # Sort by score and return top results
        scored_standards.sort(key=lambda x: x[0], reverse=True)
        return [standard for _, standard in scored_standards[:limit]]
    
    def get_compliance_report(self, 
                            organization_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate compliance report for an organization
        
        Args:
            organization_profile: Organization details and characteristics
            
        Returns:
            Dict[str, Any]: Compliance report
        """
        report = {
            "organization": organization_profile.get("name", "Unknown"),
            "generated_at": datetime.now().isoformat(),
            "applicable_standards": [],
            "compliance_status": {},
            "recommendations": [],
            "risk_assessment": {}
        }
        
        # Determine applicable standards based on organization profile
        applicable_standards = []
        
        # Industry-based filtering
        industry = organization_profile.get("industry", "").lower()
        jurisdiction = organization_profile.get("jurisdiction", "").lower()
        
        for standard in self.standards_cache.values():
            is_applicable = False
            
            # Check industry relevance
            if industry and any(industry in category for category in standard.categories):
                is_applicable = True
            
            # Check jurisdiction
            if jurisdiction and standard.jurisdiction.value == jurisdiction:
                is_applicable = True
            
            # Check organization size/type
            org_size = organization_profile.get("size", "").lower()
            if org_size == "large" and standard.compliance_level == ComplianceLevel.MANDATORY:
                is_applicable = True
            
            if is_applicable:
                applicable_standards.append(standard)
        
        report["applicable_standards"] = [
            {
                "id": s.id,
                "title": s.title,
                "compliance_level": s.compliance_level.value,
                "effective_date": s.effective_date.isoformat(),
                "requirements": s.requirements
            }
            for s in applicable_standards
        ]
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(applicable_standards, organization_profile)
        
        return report
    
    def _generate_recommendations(self, 
                                standards: List[Standard], 
                                profile: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Check for mandatory standards
        mandatory_standards = [s for s in standards if s.compliance_level == ComplianceLevel.MANDATORY]
        if mandatory_standards:
            recommendations.append(f"Priority: Implement {len(mandatory_standards)} mandatory standards")
        
        # Check for upcoming deadlines
        upcoming_deadlines = [
            s for s in standards 
            if s.effective_date > datetime.now() and s.effective_date < datetime.now() + timedelta(days=90)
        ]
        if upcoming_deadlines:
            recommendations.append(f"Urgent: {len(upcoming_deadlines)} standards take effect within 90 days")
        
        # Industry-specific recommendations
        industry = profile.get("industry", "").lower()
        if industry == "healthcare":
            recommendations.append("Consider HIPAA compliance assessment")
        elif industry == "financial":
            recommendations.append("Review PCI DSS requirements")
        elif industry == "technology":
            recommendations.append("Implement data privacy framework")
        
        return recommendations
    
    def export_standards(self, 
                        format: str = "json", 
                        output_path: Optional[str] = None,
                        filters: Optional[Dict[str, Any]] = None) -> str:
        """
        Export standards data
        
        Args:
            format: Export format ("json", "csv", "yaml")
            output_path: Optional output file path
            filters: Optional filters to apply
            
        Returns:
            str: Exported data
        """
        # Apply filters if provided
        standards = list(self.standards_cache.values())
        if filters:
            standards = self._filter_standards(**filters)
        
        # Prepare export data
        export_data = {
            "metadata": {
                "total_standards": len(standards),
                "exported_at": datetime.now().isoformat(),
                "filters_applied": filters
            },
            "standards": []
        }
        
        for standard in standards:
            export_data["standards"].append({
                "id": standard.id,
                "title": standard.title,
                "description": standard.description,
                "type": standard.standard_type.value,
                "compliance_level": standard.compliance_level.value,
                "jurisdiction": standard.jurisdiction.value,
                "version": standard.version,
                "effective_date": standard.effective_date.isoformat(),
                "expiration_date": standard.expiration_date.isoformat() if standard.expiration_date else None,
                "source_url": standard.source_url,
                "keywords": list(standard.keywords),
                "categories": list(standard.categories),
                "requirements": standard.requirements,
                "penalties": standard.penalties
            })
        
        # Export in requested format
        if format.lower() == "json":
            content = json.dumps(export_data, indent=2)
        elif format.lower() == "yaml":
            content = yaml.dump(export_data, default_flow_style=False)
        elif format.lower() == "csv":
            content = self._export_to_csv(export_data["standards"])
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Write to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
        
        return content
    
    def _export_to_csv(self, standards_data: List[Dict]) -> str:
        """Export standards to CSV format"""
        import csv
        import io
        
        if not standards_data:
            return ""
        
        output = io.StringIO()
        fieldnames = standards_data[0].keys()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in standards_data:
            # Convert lists to strings for CSV
            csv_row = {}
            for key, value in row.items():
                if isinstance(value, list):
                    csv_row[key] = "; ".join(str(v) for v in value)
                else:
                    csv_row[key] = value
            writer.writerow(csv_row)
        
        return output.getvalue()
    
    def lookup_workplace_safety_standards(self, topic: str) -> List[Standard]:
        """
        Look up workplace safety standards for a specific topic
        
        Args:
            topic: Safety topic (e.g., "forklift operation", "electrical safety")
            
        Returns:
            List[Standard]: Relevant safety standards
        """
        topic_lower = topic.lower()
        relevant_standards = []
        
        # Forklift-specific standards
        if any(word in topic_lower for word in ["forklift", "lift", "truck", "industrial truck"]):
            relevant_standards.extend(self._get_forklift_standards())
        
        # Electrical safety standards
        if any(word in topic_lower for word in ["electrical", "electric", "power", "voltage"]):
            relevant_standards.extend(self._get_electrical_safety_standards())
        
        # General workplace safety standards
        if any(word in topic_lower for word in ["safety", "hazard", "risk", "workplace"]):
            relevant_standards.extend(self._get_general_safety_standards())
        
        # Equipment safety standards
        if any(word in topic_lower for word in ["equipment", "machine", "tool", "machinery"]):
            relevant_standards.extend(self._get_equipment_safety_standards())
        
        # Fire safety standards
        if any(word in topic_lower for word in ["fire", "flame", "combustion", "extinguisher"]):
            relevant_standards.extend(self._get_fire_safety_standards())
        
        # If no specific matches, return general safety standards
        if not relevant_standards:
            relevant_standards.extend(self._get_general_safety_standards())
        
        return relevant_standards
    
    def _get_forklift_standards(self) -> List[Standard]:
        """Get forklift-specific safety standards"""
        return [
            Standard(
                id="osha_1910_178",
                title="Powered Industrial Trucks",
                description="OSHA standard for powered industrial trucks including forklifts",
                standard_type=StandardType.REGULATION,
                compliance_level=ComplianceLevel.MANDATORY,
                jurisdiction=Jurisdiction.US_FEDERAL,
                organization=StandardsOrganization.OSHA,
                standard_number="1910.178",
                version="1.0",
                effective_date=datetime(1971, 5, 29),
                keywords={"forklift", "industrial truck", "powered", "osha"},
                categories={"material handling", "equipment safety", "workplace safety"},
                requirements=[
                    "Operator training and certification required",
                    "Daily inspection of forklifts",
                    "Proper load handling procedures",
                    "Safe operating speeds and practices",
                    "Maintenance and repair requirements"
                ],
                penalties=[
                    "Up to $15,625 per violation",
                    "Willful violations up to $156,259",
                    "Repeat violations up to $156,259"
                ],
                source_url="https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.178"
            ),
            Standard(
                id="ansi_b56_1",
                title="Safety Standard for Low Lift and High Lift Trucks",
                description="ANSI standard for safety requirements of powered industrial trucks",
                standard_type=StandardType.FRAMEWORK,
                compliance_level=ComplianceLevel.RECOMMENDED,
                jurisdiction=Jurisdiction.US_FEDERAL,
                organization=StandardsOrganization.ANSI,
                standard_number="B56.1",
                version="2020",
                effective_date=datetime(2020, 1, 1),
                keywords={"forklift", "safety", "ansi", "industrial truck"},
                categories={"equipment safety", "material handling"},
                requirements=[
                    "Design and construction requirements",
                    "Safety features and devices",
                    "Performance and testing requirements",
                    "Marking and labeling requirements"
                ]
            ),
            Standard(
                id="niosh_forklift_guide",
                title="Preventing Injuries and Deaths of Workers Who Operate or Work Near Forklifts",
                description="NIOSH guidance for forklift safety",
                standard_type=StandardType.GUIDELINE,
                compliance_level=ComplianceLevel.RECOMMENDED,
                jurisdiction=Jurisdiction.US_FEDERAL,
                organization=StandardsOrganization.NIOSH,
                standard_number="NIOSH 2001-109",
                version="1.0",
                effective_date=datetime(2001, 1, 1),
                keywords={"forklift", "safety", "niosh", "guidance"},
                categories={"workplace safety", "equipment safety"},
                requirements=[
                    "Comprehensive operator training",
                    "Regular equipment maintenance",
                    "Safe work environment design",
                    "Incident investigation procedures"
                ]
            )
        ]
    
    def _get_electrical_safety_standards(self) -> List[Standard]:
        """Get electrical safety standards"""
        return [
            Standard(
                id="osha_1910_147",
                title="The Control of Hazardous Energy (Lockout/Tagout)",
                description="OSHA standard for controlling hazardous energy during equipment service",
                standard_type=StandardType.REGULATION,
                compliance_level=ComplianceLevel.MANDATORY,
                jurisdiction=Jurisdiction.US_FEDERAL,
                organization=StandardsOrganization.OSHA,
                standard_number="1910.147",
                version="1.0",
                effective_date=datetime(1989, 9, 1),
                keywords={"electrical", "lockout", "tagout", "energy", "hazardous"},
                categories={"electrical safety", "energy control", "workplace safety"},
                requirements=[
                    "Lockout/tagout procedures",
                    "Employee training requirements",
                    "Periodic inspections",
                    "Energy control program"
                ]
            ),
            Standard(
                id="nfpa_70e",
                title="Standard for Electrical Safety in the Workplace",
                description="NFPA standard for electrical safety requirements",
                standard_type=StandardType.FRAMEWORK,
                compliance_level=ComplianceLevel.RECOMMENDED,
                jurisdiction=Jurisdiction.US_FEDERAL,
                organization=StandardsOrganization.NFPA,
                standard_number="70E",
                version="2021",
                effective_date=datetime(2021, 1, 1),
                keywords={"electrical", "safety", "nfpa", "workplace"},
                categories={"electrical safety", "workplace safety"},
                requirements=[
                    "Electrical safety program",
                    "Risk assessment procedures",
                    "Personal protective equipment",
                    "Electrical safety training"
                ]
            )
        ]
    
    def _get_general_safety_standards(self) -> List[Standard]:
        """Get general workplace safety standards"""
        return [
            Standard(
                id="osha_1910_132",
                title="Personal Protective Equipment",
                description="OSHA standard for personal protective equipment requirements",
                standard_type=StandardType.REGULATION,
                compliance_level=ComplianceLevel.MANDATORY,
                jurisdiction=Jurisdiction.US_FEDERAL,
                organization=StandardsOrganization.OSHA,
                standard_number="1910.132",
                version="1.0",
                effective_date=datetime(1994, 4, 6),
                keywords={"ppe", "personal protective equipment", "safety"},
                categories={"workplace safety", "personal protection"},
                requirements=[
                    "PPE hazard assessment",
                    "PPE selection and use",
                    "Employee training on PPE",
                    "PPE maintenance and replacement"
                ]
            ),
            Standard(
                id="osha_1910_1200",
                title="Hazard Communication",
                description="OSHA standard for hazard communication (HazCom)",
                standard_type=StandardType.REGULATION,
                compliance_level=ComplianceLevel.MANDATORY,
                jurisdiction=Jurisdiction.US_FEDERAL,
                organization=StandardsOrganization.OSHA,
                standard_number="1910.1200",
                version="1.0",
                effective_date=datetime(2012, 3, 26),
                keywords={"hazard communication", "hazcom", "chemical safety"},
                categories={"chemical safety", "workplace safety"},
                requirements=[
                    "Safety data sheets (SDS)",
                    "Chemical labeling requirements",
                    "Employee training program",
                    "Written hazard communication program"
                ]
            )
        ]
    
    def _get_equipment_safety_standards(self) -> List[Standard]:
        """Get equipment safety standards"""
        return [
            Standard(
                id="osha_1910_212",
                title="General Requirements for All Machines",
                description="OSHA standard for general machine safety requirements",
                standard_type=StandardType.REGULATION,
                compliance_level=ComplianceLevel.MANDATORY,
                jurisdiction=Jurisdiction.US_FEDERAL,
                organization=StandardsOrganization.OSHA,
                standard_number="1910.212",
                version="1.0",
                effective_date=datetime(1971, 5, 29),
                keywords={"machine", "equipment", "safety", "guarding"},
                categories={"equipment safety", "machine safety"},
                requirements=[
                    "Machine guarding requirements",
                    "Safety device requirements",
                    "Operator training",
                    "Maintenance procedures"
                ]
            )
        ]
    
    def _get_fire_safety_standards(self) -> List[Standard]:
        """Get fire safety standards"""
        return [
            Standard(
                id="nfpa_10",
                title="Standard for Portable Fire Extinguishers",
                description="NFPA standard for portable fire extinguisher requirements",
                standard_type=StandardType.FRAMEWORK,
                compliance_level=ComplianceLevel.RECOMMENDED,
                jurisdiction=Jurisdiction.US_FEDERAL,
                organization=StandardsOrganization.NFPA,
                standard_number="10",
                version="2018",
                effective_date=datetime(2018, 1, 1),
                keywords={"fire", "extinguisher", "safety", "nfpa"},
                categories={"fire safety", "emergency response"},
                requirements=[
                    "Fire extinguisher selection",
                    "Installation requirements",
                    "Inspection and maintenance",
                    "Employee training"
                ]
            )
        ]
    
    def get_standard_by_number(self, standard_number: str) -> Optional[Standard]:
        """
        Get a specific standard by its number
        
        Args:
            standard_number: Standard number (e.g., "1910.178", "NFPA 70E")
            
        Returns:
            Optional[Standard]: The standard if found
        """
        for standard in self.standards_cache.values():
            if standard.standard_number == standard_number:
                return standard
        return None
    
    def get_standards_by_organization(self, organization: StandardsOrganization) -> List[Standard]:
        """
        Get all standards from a specific organization
        
        Args:
            organization: Standards organization
            
        Returns:
            List[Standard]: Standards from the organization
        """
        return [
            standard for standard in self.standards_cache.values()
            if standard.organization == organization
        ]


# Example usage
async def main():
    """Example usage of the Standards Lookup Pipeline"""
    async with StandardsLookupPipeline() as pipeline:
        # Fetch standards
        standards = await pipeline.fetch_standards(
            keywords=["privacy", "data protection"],
            jurisdiction=Jurisdiction.US_FEDERAL
        )
        
        print(f"Fetched {len(standards)} standards")
        
        # Search for specific standards
        search_results = pipeline.search_standards("GDPR", limit=5)
        print(f"Found {len(search_results)} GDPR-related standards")
        
        # Generate compliance report
        org_profile = {
            "name": "Example Corp",
            "industry": "technology",
            "jurisdiction": "us_federal",
            "size": "large"
        }
        
        report = pipeline.get_compliance_report(org_profile)
        print(f"Generated compliance report for {report['organization']}")
        
        # Export standards
        exported = pipeline.export_standards("json")
        print(f"Exported {len(exported)} characters of standards data")


if __name__ == "__main__":
    asyncio.run(main())
