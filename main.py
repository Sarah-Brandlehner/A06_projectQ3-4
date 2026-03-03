"""
Example
"""

import numpy as np

# main properties of the simulation
NUM_AIRCRAFT = 10

if __name__ == "__main__":
    import random
    random.seed(52)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    import time
    from tqdm import tqdm

    # RL model
    import atcenv.TempConfig as tc

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=10) # changed from 10000 to 10 for testing
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()
    #tracemalloc.start()

    # init environment
    env = Environment(**vars(args.env))

    load_models = False
    test = False

 
    # increase number of flights
    tot_rew_list = []
    conf_list = []
    speeddif_list = []
    # run episodes
    state_list = []
    for e in tqdm(range(args.episodes)):   
        print('\n-----------------------------------------------------')
        #snapshot1 = tracemalloc.take_snapshot()     
        episode_name = "EPISODE_" + str(e) 

        # reset environment
        number_of_aircraft = NUM_AIRCRAFT 
        obs = env.reset(number_of_aircraft)

        # set done status to false
        done = False

        # save how many steps it took for this episode to finish
        number_steps_until_done = 0
        # save how many conflics happened in eacj episode
        number_conflicts = 0
        # save different from optimal speed
        average_speed_dif = 0

        tot_rew = 0
        # execute one episode
        while not done:
            # default - do nothing - YOU HAVE TO CHANGE THIS!
            # 2 - heading and/or speed 
            actions = np.zeros((len(obs), 2))
            # perform step with dummy action
            obs, rew, done_t, done_e, info = env.step(actions)

            if done_t or done_e:
                done = True

            #for obs_i in obs:
            #    state_list.append(obs_i)
            tot_rew += rew

            number_steps_until_done += 1
            number_conflicts += len(env.conflicts)
            average_speed_dif = np.average([env.average_speed_dif, average_speed_dif])            
                
        if len(tot_rew_list) < 100:
            tot_rew_list.append(sum(tot_rew)/number_of_aircraft)
            conf_list.append(number_conflicts)
            speeddif_list.append(average_speed_dif)
        else:
            tot_rew_list[e%100 -1] = sum(tot_rew)/number_of_aircraft
            conf_list[e%100 -1] = number_conflicts
            speeddif_list[e%100 -1] = average_speed_dif
       
        #number_losses_of_sep = len(env.)
        # save information
        tc.dump_pickle(number_steps_until_done, 'results/save/numbersteps_' + episode_name)
        tc.dump_pickle(number_conflicts, 'results/save/numberconflicts_' + episode_name)
        # adding the two missing rows 
       # tc.dump_pickle()
        tc.dump_pickle(average_speed_dif, 'results/save/speeddif_' + episode_name)
        print(f'Done aircraft: {len(env.done)}')  
        print(f'Done aircraft IDs: {env.done}')      

        print(episode_name,'ended in', number_steps_until_done, 'runs, with', np.mean(np.array(conf_list)), 'conflicts (rolling av100), reward (rolling av100)=', np.mean(np.array(tot_rew_list)), 'speed dif (rolling av100)', np.mean(np.array(average_speed_dif)))        

    env.close()
