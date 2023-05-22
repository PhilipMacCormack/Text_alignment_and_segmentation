from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import UtilityFunction
from process import main_process
from process import get_gt_BB_len
from utils import get_gt_lines

path = '../../../../Data/Labours_Memory/export_job_2576375/776034/Örsundsbro,_Giresta_avd_025/'
file = 'fac_03008_arsberattelse_1931'

def black_box_function(t1, min_gap):
    path = '../../../../Data/Labours_Memory/export_job_2576375/776034/Örsundsbro,_Giresta_avd_025/'
    file = 'fac_03008_arsberattelse_1931'
    # path = '../data/'
    # file = 'a01-077u-cropped'
    holes = 'y'
    t1, peak_thresh = int(t1), int(min_gap)
    hor_rat = 150
    ver_rat = 27
    min_ak = 69
    max_ak = 2
    min_ad = 22
    max_height = 158
    peak_thresh = 35
    gt_words_lines = get_gt_lines(path,file)
    predicted_words_lines = main_process(path, file, holes, t1, hor_rat, ver_rat, min_ak, max_ak, min_ad, max_height, peak_thresh, min_gap)
    frac_list = []
    if predicted_words_lines == 0:
        return 0
    for i,predicted_line in enumerate(predicted_words_lines):
        frac = predicted_line/gt_words_lines[i]
        if frac > 1:
            frac = gt_words_lines[i]/predicted_line
        frac_list.append(frac)
    res_val = sum(frac_list)/len(frac_list)    
    return res_val
    # if len_predicted_BB/len_ground_truth_BB > 1:
    #     return len_ground_truth_BB/len_predicted_BB
    # else:
    #     return len_predicted_BB/len_ground_truth_BB

# p_bounds = {'t1':(100,230), 'hor_rat':(10,200), 'ver_rat':(5,60), 'min_ak':(1,100),'max_ak':(1,15),'min_ad':(3,70),'max_height':(55,250),'peak_thresh':(25,80)}
p_bounds = {'t1':(120,235), 'min_gap':(4,45)}

optimizer = BayesianOptimization(
    f=None,
    pbounds=p_bounds,
    verbose=2,
    random_state=1,
    allow_duplicate_points=True
)

# --- Logging ---
# load_logs(optimizer, logs=["./logs/bayes_logs.json"])
# logger = JSONLogger(path='./logs/bayes_logs.json', reset=False)
# optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
# print("Optimizer is now aware of {} points.".format(len(optimizer.space)))

utility = UtilityFunction(kappa=1.5)
optimizer.set_gp_params(alpha=1e-8)

# Probe at t1 = 205 and min_gap = 33, because they are reasonable parameters
first_point = {'t1':205,'min_gap':33}
print('Iteration 1 . Probing with parameters: t1 =',int(first_point['t1']), ', min_gap =', int(first_point['min_gap']), '. . . .')
first_target = black_box_function(first_point['t1'],first_point['min_gap'])
optimizer.register(params=first_point,target=first_target)
print('Target found: ', first_target)
frst = True
# Probe 15 times
for _ in range(15):
    next_point = optimizer.suggest(utility)
    print('Iteration',_+2,'. Probing with parameters: t1 =',int(next_point['t1']), ', min_gap =', int(next_point['min_gap']), '. . . .')
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)
    print('Target found: ', target)
    if target > 0.9:
        if frst:
            with open('./results/{}/parameters.txt'.format(file), 'a') as paramfile:
                paramfile.write('File: {} \n'.format(file))
            frst = False
        with open('./results/{}/parameters.txt'.format(file), 'a') as paramfile:
            paramfile.write('Target: {}, Parameters: [t1:{}, min_gap:{}] \n'.format(round(target,3), int(next_point['t1']), int(next_point['min_gap'])))
    if target == 1:
        print('')
        print('******* Found target value equal to 1 *******')
        break

max_params = optimizer.max['params']
print('Best parameters found: t1 = {}, min_gap = {}, with target = {}'.format(int(max_params['t1']), int(max_params['min_gap']), optimizer.max['target']))
with open('./results/{}/best_params.xml'.format(file), 'w') as best_params:
    best_params.write('\
    <alto> \n \
        <File>{}</File> \n \
        <Target>{}</Target> \n \
        <Parameters t1="{}" min_gap="{}" /> \n \
    </alto>'.format(file, str(round(optimizer.max['target'],3)), str(int(max_params['t1'])), str(int(max_params['min_gap']))))
