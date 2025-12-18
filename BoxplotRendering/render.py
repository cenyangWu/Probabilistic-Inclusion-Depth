

import bpy
import os
import math
import time



INPUT_GLB_FILE = "./data/glb/model.glb"
RENDER_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "final_glass_render.png")
RENDER_SAMPLES = 512 


def clear_scene():
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for block in collection:
            if block.users == 0:
                collection.remove(block)

def setup_render_engine():
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = RENDER_SAMPLES
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = RENDER_OUTPUT_PATH

    
    bpy.context.scene.cycles.max_bounces = 32
    bpy.context.scene.cycles.transparent_max_bounces = 32
    bpy.context.scene.cycles.transmission_max_bounces = 32

    
    bpy.context.scene.render.film_transparent = True
  
    bpy.context.scene.render.image_settings.view_settings.view_transform = 'Standard'


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
   

def add_studio_floor(model_center, model_size):
   
    floor_size = model_size * 5
    bpy.ops.mesh.primitive_plane_add(size=floor_size, enter_editmode=False, align='WORLD', location=(model_center[0], model_center[1], model_center[2] - model_size * 0.5))
    floor = bpy.context.object
    floor.name = "StudioFloor"
    
    mat_floor = bpy.data.materials.new(name="WhiteFloor")
    mat_floor.use_nodes = True
    nodes = mat_floor.node_tree.nodes
    nodes.clear()
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1)
    principled_bsdf.inputs['Roughness'].default_value = 0.5
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mat_floor.node_tree.links.new(principled_bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    floor.data.materials.append(mat_floor)
    



def main():
   
    start_time = time.time()
   
    clear_scene()
    setup_render_engine()

   
    if not os.path.exists(INPUT_GLB_FILE):
        print(f"Unfind: {INPUT_GLB_FILE}")
        return
    bpy.ops.import_scene.gltf(filepath=INPUT_GLB_FILE)
    imported_objects = sorted([obj for obj in bpy.data.objects if obj.name in ['mesh0', 'mesh1', 'mesh2','mesh3']], key=lambda o: o.name)
    if not imported_objects:
        
        return
    print(f"Successfully imported {len(imported_objects)} models.")

  
    layer_properties = [
        {'core_color': (0.517, 0.841, 1, 1), 'edge_color': (0.372, 0.799, 1, 1), 'fresnel_ior': 1.15},
        {'core_color': (1,0.637, 0.440, 1), 'edge_color': (1.0, 0.177, 0.096, 1), 'fresnel_ior': 1.25},
        
        {'core_color': (0.372, 0.799, 1, 1), 'edge_color': (0.244, 0.800, 1, 1), 'fresnel_ior': 1.45},
        # {'core_color': (0.900, 0.900, 0.900, 1), 'edge_color': (0.700, 0.700, 0.700, 1), 'fresnel_ior': 1.0},
        {'core_color': (0.892, 0.950, 0.370, 1), 'edge_color': (0.936, 0.949, 0.209, 1), 'fresnel_ior': 1.35},
    ]
    

    
    
    if len(imported_objects) != len(layer_properties):
        print(f"[Warning] {len(imported_objects)} models have been imported, but {len(layer_properties)} sets of properties have been defined. Please check the configuration.")
    for i, obj in enumerate(imported_objects):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        obj.data.materials.clear()
        if i < len(layer_properties):
            props = layer_properties[i]
            mat_name = f"PhysicalContour_{obj.name}"
            glass_material = create_physical_contour_material(mat_name, props['core_color'],props['edge_color'], props['fresnel_ior'])
            obj.data.materials.append(glass_material)
    
    min_coords = [min(obj.bound_box[i][j] for obj in imported_objects for i in range(8)) for j in range(3)]
    max_coords = [max(obj.bound_box[i][j] for obj in imported_objects for i in range(8)) for j in range(3)]
    model_center = tuple((min_c + max_c) / 2 for min_c, max_c in zip(min_coords, max_coords))
    model_size = max(max_coords[i] - min_coords[i] for i in range(3))
    # FIXED_REFERENCE_SCALE = 162.0
    setup_pure_white_world()
    setup_clean_studio_lighting(model_center, model_size)
    

    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=model_center)
    camera_target = bpy.context.object
    camera_target.name = "CameraTarget"
    
    camera_distance = model_size * 5.5
    # camera_distance = FIXED_REFERENCE_SCALE * 5.5
    camera_location = (model_center[0]- camera_distance , model_center[1], model_center[2] + model_size * 0.1)
    
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.data.lens = 100
    look_at_constraint = camera.constraints.new(type='TRACK_TO')
    look_at_constraint.target = camera_target
  
   
   
  

    
    print(f"The rendering result will be saved to: {RENDER_OUTPUT_PATH}")
    bpy.ops.render.render(write_still=True)
    print("--- Rendering completed! ---")

   
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Time take: {elapsed_time:.2f} s")

if __name__ == "__main__":
    main()