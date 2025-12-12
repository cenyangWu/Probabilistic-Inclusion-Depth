import vtk
import os



MODEL_LAYERS = [
  
    # {
    #     # !!! 请将这里的文件名替换成您第四个文件的实际名称 !!!
    #     'filepath': 'connected_deepest_member_density_member.nii', 
        
    #     'color': [0.2, 0.6, 1.0], 
    #     'isovalue': 0.5,
    # },
    {
        'filepath': 'all_masks_union.nii',
        'color': [0.8, 0.8, 0.95], 
        'isovalue': 0.5,
    },
    {
        'filepath': 'top50percent_depth_union.nii',
        'color': [0.4, 0.9, 0.5],
        'isovalue': 0.5,
    },
    {
        'filepath': 'top50percent_depth_intersection.nii',
        'color': [1.0, 0.4, 0.4], 
        'isovalue': 0.5,
    },
    {
      
        'filepath': 'deepest_mask_IXI038-Guys-0729_aligned_seg35_LPI.nii', 
        
        'color': [0.2, 0.6, 1.0], 
        'isovalue': 0.5,
    }

    
]

OUTPUT_3D_SCENE_NAME = "professional_smooth_model.glb"


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
 
    export_renderer = vtk.vtkRenderer()
    export_render_win = vtk.vtkRenderWindow()

    export_render_win.SetOffScreenRendering(1)
    export_render_win.AddRenderer(export_renderer)


    for layer_config in MODEL_LAYERS:
        nii_path = os.path.join(script_dir, layer_config['filepath'])
        if not os.path.exists(nii_path):
            print(f"  -[Warning] File not found, skipping: {nii_path}")
            continue

        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(nii_path)

        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputConnection(reader.GetOutputPort())
        marching_cubes.SetValue(0, layer_config['isovalue'])
        marching_cubes.Update()
        
        polydata = marching_cubes.GetOutput()
   
        if polydata.GetNumberOfPoints() == 0:
            print(f"  -[Warning] The surface could not be extracted from the file {layer_config['filepath']}")
            continue

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(polydata)
        smoother.SetNumberOfIterations(30) 
        smoother.SetPassBand(0.1)          
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()

        smooth_polydata = smoother.GetOutput()
        
        # center = smooth_polydata.GetCenter()
        # transform = vtk.vtkTransform()
        # transform.Translate(-center[0], -center[1], -center[2])

        # transform_filter = vtk.vtkTransformPolyDataFilter()
        # transform_filter.SetInputData(smooth_polydata)
        # transform_filter.SetTransform(transform)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(layer_config['color'])
       
        export_renderer.AddActor(actor)
 
    exporter = vtk.vtkGLTFExporter()
    scene_output_path = os.path.join(script_dir, OUTPUT_3D_SCENE_NAME)
    exporter.SetFileName(scene_output_path)
    exporter.SetRenderWindow(export_render_win)
    exporter.Write()
    
    print(f"--- The high-quality smooth model has been saved to: {scene_output_path} ---")
    

    bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    actors = export_renderer.GetActors()

    if actors.GetNumberOfItems() == 0:
        print("SIZE:0.0")
        return

    bounds = [1e20, -1e20, 1e20, -1e20, 1e20, -1e20]

    actors.InitTraversal()
    actor = actors.GetNextActor()
    while actor:
        actor_bounds = actor.GetBounds()
        bounds[0] = min(bounds[0], actor_bounds[0])
        bounds[1] = max(bounds[1], actor_bounds[1])
        bounds[2] = min(bounds[2], actor_bounds[2])
        bounds[3] = max(bounds[3], actor_bounds[3])
        bounds[4] = min(bounds[4], actor_bounds[4])
        bounds[5] = max(bounds[5], actor_bounds[5])
        actor = actors.GetNextActor()

    model_size_x = bounds[1] - bounds[0]
    model_size_y = bounds[3] - bounds[2]
    model_size_z = bounds[5] - bounds[4]
    model_size = max(model_size_x, model_size_y, model_size_z)

    print(f"--- Model Dimensions (X, Y, Z): ({model_size_x:.2f}, {model_size_y:.2f}, {model_size_z:.2f}) ---")
    print(f"SIZE:{model_size}")
    

if __name__ == "__main__":
    main()