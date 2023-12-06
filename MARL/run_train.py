
import os


def run(model, config_file, base_dir):
    py = "python "
    
    script = "run_"+model+".py --option train "

    ## config
    config = "--config-dir "+config_file+" "

    ## base dir
    base = " --base-dir "+base_dir+" "

    command = py + script + config + base
    print(command)
    # os.system(command)


## MAA2C   low_traffic
run("ma2c", "configs/configs.ini", "./MAA2C_low_res/")
## MAA2C   meduim_traffic
run("ma2c", "configs/configs_med.ini", "./MAA2C_med_res/")
## MAPPO    low_traffic
run("mappo", "configs/configs_ppo.ini", "./MAPPO_low_res/")
## MAPPO    medium_traffic
run("mappo", "configs/configs_ppo_med.ini", "./MAPPO_med_res/")
