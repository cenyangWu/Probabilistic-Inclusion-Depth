import os
import subprocess


parent_folder = './ScalarFlowData' 

output_folder = './output' 

vtk_to_glb_script = 'vtk_to_glb.py'

render_blender_script = 'render.py'

render_video_script = 'render_video.py'


os.makedirs(output_folder, exist_ok=True)
target_folder_name = ['model5'] 


# for folder_name in os.listdir(parent_folder):


# folder_name = target_folder_name



for folder_name in target_folder_name:
# for folder_name in os.listdir(parent_folder):
    sub_folder_path = os.path.join(parent_folder, folder_name)
  
    if not os.path.isdir(sub_folder_path):
        continue
    
    output_glb_name = f"{folder_name}_model.glb"
    glb_path = os.path.join(sub_folder_path, output_glb_name).replace(os.sep, '/')

    temp_vtk_script_path = os.path.join(sub_folder_path, "temp_vtk_to_glb.py")
    try:
        with open(vtk_to_glb_script, 'r', encoding='utf-8') as f_in:
            vtk_script_content = f_in.read()
        
        vtk_script_content = vtk_script_content.replace(
            "script_dir = os.path.dirname(os.path.realpath(__file__))",
            f"script_dir = '{sub_folder_path.replace(os.sep, '/')}'"
        )
       
        vtk_script_content = vtk_script_content.replace(
            'OUTPUT_3D_SCENE_NAME = "professional_smooth_model.glb"',
            f'OUTPUT_3D_SCENE_NAME = "{output_glb_name}"'
        )
      
        with open(temp_vtk_script_path, 'w', encoding='utf-8') as f_out:
            f_out.write(vtk_script_content)

        print("1. Converting .nii to .glb...")
       
        subprocess.run(['python', temp_vtk_script_path], check=True, capture_output=True, text=True)
        print("  Conversion successful.")
        
    except FileNotFoundError:
        print(f" [Error] Script or file not found, please check the path configuration.")
        continue 
    except subprocess.CalledProcessError as e:
        print(f"  [Error] The conversion process from .nii to .glb failed:{e}")
        
        continue 
    finally:
     
        if os.path.exists(temp_vtk_script_path):
            os.remove(temp_vtk_script_path)

    
    temp_blender_script_path = os.path.join(sub_folder_path, "temp_render_blender.py")
    try:
        with open(render_blender_script, 'r', encoding='utf-8') as f_in:
            blender_script_content = f_in.read()

      
        blender_script_content = blender_script_content.replace(
            'INPUT_GLB_FILE = "/home/guoshudan/Documents/internProject/glb1/professional_smooth_model.glb"',
            f'INPUT_GLB_FILE = "{glb_path}"'
        )
        
        render_output_name = f"{folder_name}_render.png"
        render_output_path = os.path.join(output_folder, render_output_name).replace(os.sep, '/')
        blender_script_content = blender_script_content.replace(
            'RENDER_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "final_glass_render.png")',
            f'RENDER_OUTPUT_PATH = "{render_output_path}"'
        )
        
        with open(temp_blender_script_path, 'w', encoding='utf-8') as f_out:
            f_out.write(blender_script_content)

        print("Currently using Blender to render .glb into images...")
      
        subprocess.run(['blender', '--background', '--python', temp_blender_script_path], check=True, capture_output=True, text=True)
        print(" The image rendering is successful.")
        
    except FileNotFoundError:
        print(f"   [Error] Blender program or rendering script not found, please check the path.")
        continue
    except subprocess.CalledProcessError as e:
        print(f"  Fail: {e}")
        print(f" Error details: {e.stderr}")
        continue
    finally:
        if os.path.exists(temp_blender_script_path):
            os.remove(temp_blender_script_path)

    
    temp_video_script_path = os.path.join(sub_folder_path, "temp_render_video.py")
    try:
        with open(render_video_script, 'r', encoding='utf-8') as f_in:
            video_script_content = f_in.read()

        video_script_content = video_script_content.replace(
            'INPUT_GLB_FILE = "/home/guoshudan/Documents/internProject/glb1/professional_smooth_model.glb"',
            f'INPUT_GLB_FILE = "{glb_path}"'
        )
        

        video_output_name = f"{folder_name}_video.mp4"
        video_output_path = os.path.join(output_folder, video_output_name).replace(os.sep, '/')
        video_script_content = video_script_content.replace(
            'VIDEO_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "final_rotation_video2.mp4")',
            f'VIDEO_OUTPUT_PATH = "{video_output_path}"'
        )
        
        with open(temp_video_script_path, 'w', encoding='utf-8') as f_out:
            f_out.write(video_script_content)

        print("Currently using Blender to render .glb into a video...")
       
        subprocess.run(['blender', '--background', '--python', temp_video_script_path], check=True, capture_output=True, text=True)
        print("   The video rendering was successful.")
        
    except FileNotFoundError:
        print(f" [Error] Blender program or video rendering script not found, please check the path.")
        continue
    except subprocess.CalledProcessError as e:
        print(f"  Fail: {e}")
        print(f"  Fail detail: {e.stderr}")
        continue
    finally:
        if os.path.exists(temp_video_script_path):
            os.remove(temp_video_script_path)

print("\n--- Done successfully  ---")