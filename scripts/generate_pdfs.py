from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pathlib import Path
import random

ACTIONS = ['grasp', 'move', 'pick', 'place', 'rotate', 'push', 'pull']
OBJECTS = ['cube', 'sphere', 'cylinder', 'tool', 'gripper', 'glass_cup', 'plastic_cup', 'wooden_box', 'metal_can', 'book', 'bottle', 'toy_car']
SPATIALS = ['above', 'below', 'left', 'right', 'front', 'behind']
SAFETYS = ['collision', 'force', 'limit', 'safe', 'danger', 'overload', 'alarm', 'warning']

def generate_sentence():
    action = random.choice(ACTIONS)
    obj = random.choice(OBJECTS)
    spatial = random.choice(SPATIALS)
    safety = random.choice(SAFETYS)
    templates = [
        f"The robotic system is required to {action} the {obj} {spatial} the workspace to ensure {safety} compliance.",
        f"During operation, always {action} the {obj} when it is {spatial} to avoid {safety} events.",
        f"Automatic {action} of the {obj} {spatial} the table is recommended for {safety} reasons.",
        f"Operators must verify that the {obj} is {action}ed {spatial} the robot arm to prevent {safety}.",
        f"Routine maintenance includes {action}ing the {obj} {spatial} the storage area, minimizing {safety} risks.",
        f"Sensor feedback indicates that {action}ing the {obj} {spatial} the shelf can trigger {safety} warnings.",
        f"Before starting, check if the {obj} is ready to be {action}ed {spatial} the base to maintain {safety}.",
        f"System logs show a {safety} alert when attempting to {action} the {obj} {spatial} the conveyor.",
        f"To comply with safety protocols, the {obj} should only be {action}ed {spatial} the workspace.",
        f"Unexpected {safety} was detected while the robot tried to {action} the {obj} {spatial} the platform.",
        f"Ensure the {obj} is not {action}ed {spatial} the hazardous zone to avoid {safety}.",
        f"After each cycle, the {obj} must be {action}ed {spatial} the docking station for {safety} checks.",
        f"Failure to {action} the {obj} {spatial} the robot may result in {safety} incidents.",
        f"Periodic system diagnostics require {action}ing the {obj} {spatial} the maintenance area.",
        f"Operators are advised to {action} the {obj} {spatial} the assembly line to reduce {safety} probability.",
        f"System will automatically {action} the {obj} {spatial} the workspace if {safety} is detected.",
        f"Manual override allows the user to {action} the {obj} {spatial} the robot base during {safety} events.",
        f"Documentation recommends {action}ing the {obj} {spatial} the storage rack for optimal {safety}.",
        f"Visual inspection is necessary after {action}ing the {obj} {spatial} the platform.",
        f"Emergency stop is triggered if the {obj} is {action}ed {spatial} the danger zone."
    ]
    return random.choice(templates)

def generate_pdf(file_path, title, content_lines):
    c = canvas.Canvas(str(file_path), pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(80, height - 80, title)
    c.setFont("Helvetica", 12)
    y = height - 120
    for idx, line in enumerate(content_lines):
        c.drawString(80, y, line)
        y -= 20
        # 换页逻辑：每页最多写28行
        if (idx + 1) % 28 == 0:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 80
    c.save()

def main():
    output_dir = Path('./raw_data/pdfs')
    output_dir.mkdir(parents=True, exist_ok=True)
    num_files = 15

    for i in range(num_files):
        title = f"Robot Operation Technical Report #{i+1}"
        # 每个PDF生成60~70句，确保至少两页
        num_sentences = random.randint(60, 70)
        content_lines = [generate_sentence() for _ in range(num_sentences)]
        file_path = output_dir / f"franka_manual_{i+1}.pdf"
        generate_pdf(file_path, title, content_lines)
        print(f"生成: {file_path}")

if __name__ == '__main__':
    main()