import os
import yaml

# Define the directory containing the figures
figures_dir = r'G:\AlexUniv\alex-univ-4.2\projects\mtk-bachelor-project\website\mtk-bachelor-project-website\data\writings\posts\2024-06-24-chapter-1-trimming -AI\figures'

# Check if the figures directory exists
if not os.path.exists(figures_dir):
    print(f"Error: The directory '{figures_dir}' does not exist.")
    exit(1)

# Get a list of all .png files in the figures directory
figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]

# Create the structure for the gallery.yml file
gallery = [
    {
        'category': 'Figures',
        'description': 'Gallery of Figures',
        'tiles': [
            {
                'title': os.path.splitext(fig)[0],  # Use the filename without extension as the title
                'subtitle': f'Description of {os.path.splitext(fig)[0]}',
                'thumbnail': f'figures/{fig}'
            } for fig in figures
        ]
    }
]

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save the structure to gallery.yml in the same directory as the script
output_path = os.path.join(script_dir, 'gallery.yml')
with open(output_path, 'w') as file:
    yaml.dump(gallery, file, default_flow_style=False)

print(f"gallery.yml has been generated successfully at {output_path}.")
