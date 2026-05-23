# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


''' example/theoretical configurations
terminology:
- plate: 2D thermal medium
- block: 3D thermal medium
- grid: sampling grid projected onto medium
- axis: [width(x),height(y),depth(z)]
- sides:
    - left: x=x_min, right: x=x_max
    - bottom: y=y_min, top: y=y_max
    - back: z=z_min, front: z=z_max


'''

def single_plate_example():
    # define main ThermalSystem config as Dict
    # main dictionary is for attribs shared across all instances of a plate
    # for initial projects, size, material, and grid shape will be shared across all plates
    single2D_fixed = dict(
        grid_size=[10,10],      # num intervals in point grid for prediction
        medium_size=[1,1.5],    # [x,y] in cm, nm, ect
        medium_conduct=0.1,     # internal conductivity of material
        num_sources=1,                 # max number of power sources per plate in training
        power_source_type="gaussian",  # type of power source applied to plate
        sides=dict(
            top="ambient", bottom="ambient",
            left="touching",
            right=dict(
                type="connection", # homogeneous connection: any surface that touches w same conduct will be considered fused to this plate
                # if touched by nothing ...
                # CHANGE CONNECTIONS TO LAYOUT CLASS
            )
        )
    )

    single2D_eval_config = dict(
        source_type="gaussian",
        sources=[          # if none, will randomly select
            dict(
                source_power=0.1,   # power source measure (volts or amps or whatever)
                source_spread=0.1,  # gaussian radius
                grid_location=[1,1]    # grid location of power center
            )
        ]
    )

''' for future consideration, not implemented 
looks at model for silicone gel pad centered on right side of flat copper plate
'''
def multi_dynamic_plate_example():
    copper = dict(
        name="copper",
        grid_size=[9,9],      # num intervals in point grid for prediction
        medium_size=[0.3,0.3],    # [x,y] in cm, nm, ect
        medium_conduct=0.1,     # internal conductivity of material
        power_source_type="gaussian",  # type of power source applied to plate
        num_sources=1,                 # max number of power sources per plate in training
        #side
    )

    silicone = dict(
        name="silicone",
        grid_size=[3,3],      # num intervals in point grid for prediction
        medium_size=[0.3,0.3],    # [x,y] in cm, nm, ect
        medium_conduct=0.9,     # internal conductivity of material
        num_sources=0,          # max number of power sources per plate in training
    )

    connection = {
        "copper":"right",
        "silicone":"left"
    }

