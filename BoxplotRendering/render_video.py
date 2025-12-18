import bpy
import os
import math
import time


INPUT_GLB_FILE = "./data/glb/model.glb"
VIDEO_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "final_rotation_video2.mp4")
RENDER_SAMPLES = 256
FRAME_RATE = 24
DURATION_SECONDS = 5



def clear_scene():
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for block in collection:
            if block.users == 0:
                collection.remove(block)

def setup_render_engine_for_video():
    
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_denoising = True
    
    
    scene.render.image_settings.view_settings.view_transform = 'Standard'
    
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    
    scene.render.filepath = VIDEO_OUTPUT_PATH
    
    scene.cycles.max_bounces = 32
    scene.cycles.transparent_max_bounces = 32
    scene.cycles.transmission_max_bounces = 32

    scene.frame_start = 1
    scene.frame_end = DURATION_SECONDS * FRAME_RATE
    scene.render.fps = FRAME_RATE

def create_physical_contour_material(name, core_color, edge_color, fresnel_ior):
  
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    core_glass = nodes.new(type='ShaderNodeBsdfGlass')
    core_glass.inputs['Color'].default_value = core_color
    core_glass.inputs['Roughness'].default_value = 0.1
    core_glass.inputs['IOR'].default_value = 1.0
    edge_glass = nodes.new(type='ShaderNodeBsdfGlass')
    edge_glass.inputs['Color'].default_value = edge_color
    edge_glass.inputs['Roughness'].default_value = 0.02
    edge_glass.inputs['IOR'].default_value = 1.1
    fresnel = nodes.new(type='ShaderNodeFresnel')
    fresnel.inputs['IOR'].default_value = fresnel_ior
    links = mat.node_tree.links
    links.new(fresnel.outputs['Fac'], mix_shader.inputs['Fac'])
    links.new(core_glass.outputs['BSDF'], mix_shader.inputs[1])
    links.new(edge_glass.outputs['BSDF'], mix_shader.inputs[2])
    links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])
    return mat

def setup_pure_white_world():
   
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    nodes.clear()
    output = nodes.new(type='ShaderNodeOutputWorld')
    background = nodes.new(type='ShaderNodeBackground')
    background.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1)
    background.inputs['Strength'].default_value = 6.0  
    links = node_tree.links
    links.new(background.outputs['Background'], output.inputs['Surface'])
    print("A pure white world background has been set.")

def setup_clean_studio_lighting(model_center, model_size):
   
    light_distance = model_size * 2
    key_light_loc = (model_center[0] - light_distance, model_center[1] - light_distance, model_center[2] + light_distance * 0.8)
    bpy.ops.object.light_add(type='AREA', radius=model_size, location=key_light_loc)
    key_light = bpy.context.object.data
    key_light.energy = 200 * (model_size / 100)**2
    key_light.shape = 'DISK'
    fill_light_loc = (model_center[0] + light_distance, model_center[1] - light_distance * 0.5, model_center[2] + light_distance * 0.5)
    bpy.ops.object.light_add(type='AREA', radius=model_size * 1.5, location=fill_light_loc)
    fill_light = bpy.context.object.data
    fill_light.energy = 100 * (model_size / 100)**2
    fill_light.shape = 'DISK'
    back_light_loc = (model_center[0], model_center[1] + light_distance, model_center[2])
    bpy.ops.object.light_add(type='AREA', radius=model_size, location=back_light_loc)
    back_light = bpy.context.object.data
    back_light.energy = 150 * (model_size / 100)**2
    back_light.shape = 'DISK'
    


def main():
    
    start_time = time.time()
    print("--- 1. Initialize the scene ---")
    clear_scene()
    setup_render_engine_for_video()

    
    if not os.path.exists(INPUT_GLB_FILE):
        print(f"Unfind: {INPUT_GLB_FILE}")
        return
    bpy.ops.import_scene.gltf(filepath=INPUT_GLB_FILE)
    imported_objects = sorted([obj for obj in bpy.data.objects if obj.name in ['mesh0', 'mesh1', 'mesh2']], key=lambda o: o.name)
    if not imported_objects:
        
        return
    print(f"Successfully imported {len(imported_objects)} models.")

    
    layer_properties = [
         {'core_color': (0.517, 0.841, 1, 1), 'edge_color': (0.372, 0.799, 1, 1), 'fresnel_ior': 1.25},
        {'core_color': (1,0.637, 0.440, 1), 'edge_color': (1.0, 0.177, 0.096, 1), 'fresnel_ior': 1.35},
        {'core_color': (0.372, 0.799, 1, 1), 'edge_color': (0.244, 0.800, 1, 1), 'fresnel_ior': 1.45}
    ]
    for i, obj in enumerate(imported_objects):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        obj.data.materials.clear()
        if i < len(layer_properties):
            props = layer_properties[i]
            mat_name = f"PhysicalContour_{obj.name}"
            glass_material = create_physical_contour_material(mat_name, props['core_color'], props['edge_color'], props['fresnel_ior'])
            obj.data.materials.append(glass_material)
            
    
    
    min_coords = [min(obj.bound_box[i][j] for obj in imported_objects for i in range(8)) for j in range(3)]
    max_coords = [max(obj.bound_box[i][j] for obj in imported_objects for i in range(8)) for j in range(3)]
    model_center = tuple((min_c + max_c) / 2 for min_c, max_c in zip(min_coords, max_coords))
    model_size = max(max_coords[i] - min_coords[i] for i in range(3))

    
    setup_pure_white_world()
    setup_clean_studio_lighting(model_center, model_size)
    
    
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=model_center)
    target = bpy.context.object
    target.name = "AnimationTarget"

   
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=model_center)
    pivot = bpy.context.object
    pivot.name = "CameraPivot"

    
    camera_distance = model_size * 6.5
    camera_location = (model_center[0], model_center[1] - camera_distance, model_center[2] + camera_distance * 0.1)
    
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.data.lens = 100
    camera.data.clip_end = camera_distance * 5

   
    look_at_constraint = camera.constraints.new(type='TRACK_TO')
    look_at_constraint.target = target
    
  
    camera.parent = pivot

   
    scene = bpy.context.scene
    pivot.rotation_euler = (0, 0, 0)
    pivot.keyframe_insert(data_path="rotation_euler", frame=scene.frame_start)
    
    pivot.rotation_euler = (0, 0, math.radians(360))
    pivot.keyframe_insert(data_path="rotation_euler", frame=scene.frame_end)

    for fcurve in pivot.animation_data.action.fcurves:
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'LINEAR'

   
    
    print(f"The rendering result will be saved to: {VIDEO_OUTPUT_PATH}")
    bpy.ops.render.render(animation=True)
    print("--- Animation rendering completed! ---")

    
    end_time = time.time()

   
    elapsed_time = end_time - start_time
    print(f"Rendering completed, time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()