import time
import datetime

ai_safety_agent_test = {
    "title": "AI_Safety_Agent",
    "scenario": (
        "You are a new safety manager at a manufacturing plant. Your goal is to use the 'AI Safety Agent' "
        "to review a training module you've built to make sure it's safe and compliant before you send it to new hires."
    ),
    "tasks": [
        "Task 1 (Navigation/Goal Clarity): You have just opened the application. Find the button or link to start a new safety review and begin the process.",
        "Task 2 (Completion Flow): Upload the attached 'Forklift_Safety_Module.docx' file to be analyzed by the AI.",
        "Task 3 (Feedback Clarity): The AI has finished its review and has flagged 3 potential issues. Find the first issue and read the AI's feedback. What is it telling you to do?",
        "Task 4 (Feedback Clarity): The AI has flagged a step as 'High-Risk.' Find out *why* it's considered high-risk and see if it provides an OSHA reference.",
        "Task 5 (Ease of Use): Accept the AI's suggestion for the first issue, but 'ignore' the suggestion for the second issue.",
        "Task 6 (Completion Flow): Now that you have reviewed the feedback, find out how to approve the document and export the final, compliant version."
    ]
}

immersive_builder_test = {
    "title": "Immersive_Preview_Builder",
    "scenario": (
        "You are a course creator at the plant. You've just finished writing a text script for a new training module, "
        "and you want to use the 'Immersive Preview Builder' to turn that script into an interactive, 3D simulation for new hires."
    ),
    "tasks": [
        "Task 1 (Navigation/Goal Clarity): You have opened the builder. Start a new project for your 'Quality Inspection Training' module.",
        "Task 2 (Completion Flow): Your first step is to create the main training environment. Find the option to add a new scene or environment.",
        "Task 3 (Ease of Use): For this scene, select the 'Assembly Line' environment template from the asset library.",
        "Task 4 (Ease of Use): Now, add two objects to the scene: a 'Conveyor Belt' and a 'Quality Control Station'.",
        "Task 5 (Completion Flow): This module needs an instructor. Add the 'Floor Supervisor' avatar to the scene and place them next to the QC station.",
        "Task 6 (Goal Clarity): You want to see what your module looks like. Find the 'Preview' button and 'play' the simulation you've just built.",
        "Task 7 (Feedback Clarity/Ease of Use): After the preview, you decide the lighting is too dark. Find the environment settings and change the time of day from 'Night' to 'Day'.",
        "Task 8 (Completion Flow): You are satisfied with the preview. Find the button to 'Publish' or 'Export' your module."
    ]
}

def run_usability_test(script_data):
    """
    Runs a guided usability test session and saves the observer's notes.
    """
    all_notes = []
    
    print("="*60)
    print(f"Starting Test: {script_data['title']}")
    print("="*60)
    
    print("\n--- SCENARIO FOR TESTER ---")
    print(script_data['scenario'])
    print("\n(Read the scenario to the tester, then press Enter to begin the first task.)")
    input()
    
    for i, task in enumerate(script_data['tasks']):
        print("\n" + "-"*30)
        print(f"Task {i + 1} of {len(script_data['tasks'])}")
        print("-" * 30)
        
        print(f"\nINSTRUCT THE TESTER:\n'{task}'")
        
        print("\n(Observing tester... Press Enter when they have completed or given up on the task.)")
        input()
        
        print("--- OBSERVER NOTES ---")
        notes = input("Note friction points, quotes, completion time, etc.:\n> ")
        all_notes.append(f"Task {i + 1}: {task}\nObserver Notes: {notes}\n")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    tester_id = input("Enter a Tester ID or Name (e.g., 'internal_01' or 'jane_d'): ")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"test_results_{script_data['title']}_{tester_id}_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write(f"Test Session: {script_data['title']}\n")
        f.write(f"Tester ID: {tester_id}\n")
        f.write(f"Date/Time: {timestamp}\n")
        f.write("\n--- SCENARIO ---\n")
        f.write(script_data['scenario'] + "\n")
        f.write("\n--- OBSERVATIONS ---" + "\n")
        for note in all_notes:
            f.write(note + "-"*20 + "\n")
            
    print(f"\n✅ Success! Your notes have been saved to: {filename}")


if __name__ == "__main__":
    print("Which usability test script do you want to run?")
    print("1: AI Safety Agent")
    print("2: Immersive Preview Builder")
    
    choice = input("Enter 1 or 2: > ")
    
    if choice == '1':
        run_usability_test(ai_safety_agent_test)
    elif choice == '2':
        run_usability_test(immersive_builder_test)
    else:
        print("Invalid choice. Please run the script again and enter 1 or 2.")