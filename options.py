import os
import argparse
import yaml
import numpy as np
import datetime

_CLASS_NAME = {
    "THUMOS14": [
        'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
        'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
        'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow',
        'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing',
        'ThrowDiscus', 'VolleyballSpiking'
    ],
    "GTEA":['take', 'open', 'scoop', 'close', 'put', 'pour', 'stir'],
    "BEOID":['pick-up_plug', 'plug_plug', 'switch-on_socket', 'pick-up_tape',
       'scan_card-reader', 'open_door', 'pull_drawer', 'push_drawer',
       'press_button', 'pick-up_cup', 'turn_tap', 'rinse_cup', 'take_cup',
       'put_cup', 'pick-up_jar', 'put_jar', 'open_jar', 'take_spoon',
       'stir_spoon', 'insert_wire', 'place_tape', 'scoop_spoon',
       'pour_spoon', 'fill_cup', 'move_rest', 'move_seat',
       'pull-out_weight-pin', 'insert_weight-pin',
       'rotate_weight-setting', 'pull_rowing-machine',
       'push_rowing-machine', 'let-go_rowing-machine', 'hold-down_button',
       'insert_foot'],
    "ActivityNet1.3":['Drinking beer', 'Dodgeball', 'Doing fencing', 'Playing congas',
       'River tubing', 'Changing car wheel', 'Rock-paper-scissors',
       'Knitting', 'Removing ice from car', 'Shoveling snow',
       'Tug of war', 'Shot put', 'Baking cookies', 'Doing crunches',
       'Baton twirling', 'Slacklining', 'Painting furniture', 'Archery',
       'Snow tubing', 'Wakeboarding', 'Ballet', 'Cleaning sink',
       'Disc dog', 'Curling', 'Playing badminton', 'Making an omelette',
       'Hanging wallpaper', 'Playing accordion', 'Rafting', 'Spinning',
       'Throwing darts', 'Playing pool', 'Getting a tattoo', 'Sailing',
       'Playing bagpipes', 'Fun sliding down', 'Smoking hookah',
       'Canoeing', 'Getting a haircut', 'Calf roping', 'Kayaking',
       'Horseback riding', 'Using the pommel horse', 'Bathing dog',
       'Rope skipping', 'Smoking a cigarette', 'Windsurfing',
       'Using the balance beam', 'Chopping wood', 'Arm wrestling',
       'Powerbocking', 'Putting on makeup', 'Starting a campfire',
       'Welding', 'Futsal', 'Shaving', 'Playing flauta',
       'Playing rubik cube', 'Painting', 'Playing lacrosse',
       'Playing piano', 'Longboarding', 'Drinking coffee',
       'Using the rowing machine', 'Making a lemonade',
       'Using parallel bars', 'Fixing the roof', 'Javelin throw',
       'Rollerblading', 'Elliptical trainer', 'Bullfighting',
       'Doing a powerbomb', 'Beer pong', 'Walking the dog',
       'Clean and jerk', 'Grooming horse', 'Hitting a pinata',
       'Braiding hair', 'Grooming dog', 'Peeling potatoes',
       'Vacuuming floor', 'Playing squash', 'Having an ice cream',
       'Tai chi', 'Playing harmonica', 'Swinging at the playground',
       'Camel ride', 'Triple jump', 'Doing kickboxing', 'Laying tile',
       'Springboard diving', 'Skiing', 'Decorating the Christmas tree',
       'Applying sunscreen', 'High jump', 'Preparing pasta',
       'Gargling mouthwash', 'Playing ten pins', 'Spread mulch',
       'Plastering', 'Drum corps', 'Doing step aerobics', 'Surfing',
       'Blowing leaves', 'Snowboarding', 'Playing drums', 'Skateboarding',
       'BMX', 'Raking leaves', 'Cleaning shoes', 'Beach soccer',
       'Ice fishing', 'Playing blackjack', 'Waterskiing', 'Waxing skis',
       'Belly dance', 'Getting a piercing', 'Doing nails',
       'Tennis serve with ball bouncing', 'Discus throw',
       'Mowing the lawn', 'Hand washing clothes', 'Wrapping presents',
       'Playing guitarra', 'Playing water polo', 'Hammer throw',
       'Roof shingle removal', 'Blow-drying hair',
       'Playing beach volleyball', 'Sumo', 'Cheerleading',
       'Bungee jumping', 'Making a cake', 'Rock climbing', 'Hopscotch',
       'Cutting the grass', 'Layup drill in basketball', 'Washing face',
       'Playing violin', 'Sharpening knives', 'Polishing forniture',
       'Ping-pong', 'Mixing drinks', 'Table soccer', 'Playing kickball',
       'Kite flying', 'Playing ice hockey', 'Building sandcastles',
       'Playing polo', 'Doing karate', 'Installing carpet',
       'Running a marathon', 'Painting fence', 'Cleaning windows',
       'Riding bumper cars', 'Ironing clothes', 'Croquet', 'Cumbia',
       'Making a sandwich', 'Capoeira', 'Putting in contact lenses',
       'Brushing teeth', 'Preparing salad', 'Tumbling',
       'Playing field hockey', 'Trimming branches or hedges', 'Long jump',
       'Brushing hair', 'Washing dishes', 'Kneeling', 'Hurling',
       'Hula hoop', 'Washing hands', 'Using the monkey bar',
       'Using uneven bars', 'Hand car wash', 'Mooping floor',
       'Scuba diving', 'Zumba', 'Putting on shoes', 'Polishing shoes',
       'Assembling bicycle', 'Shaving legs', 'Swimming',
       'Clipping cat claws', 'Shuffleboard', 'Volleyball', 'Breakdancing',
       'Paintball', 'Carving jack-o-lanterns', 'Snatch', 'Tango',
       'Cricket', 'Doing motocross', 'Pole vault', 'Playing racquetball',
       'Plataform diving', 'Fixing bicycle', 'Playing saxophone',
       'Removing curlers'],
}

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def parse_args():
    parser = argparse.ArgumentParser("Official Pytorch Implementation of HR-Pro: Point-supervised Temporal Action Localization \
                                        via Hierarchical Reliability Propagation")
    
    parser.add_argument('--cfg', type=str, default='thumos', help='hyperparameters path')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'infer'])
    parser.add_argument('--stage', type=int, default=1, help='traning stage', choices=[1, 2])
    parser.add_argument('--seed', type=int, default=0, help='random seed (-1 for no manual seed)')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='root folder for saving models,ouputs,logs')
    #
    parser.add_argument("--mtl", help='multi_task_learning', action="store_true")
    # parser.add_argument('--mtl_task', type=str, default=1, help='specific task',
    #                     choices=['pred','order'])
    args = parser.parse_args()

    # hyper-params from ymal file
    with open('./cfgs/{}_hyp.yaml'.format(args.cfg)) as f:
        hyp_dict = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in hyp_dict.items():
        setattr(args, key, value)

    #
    if args.mtl:
        args.model_previous_s1 = os.path.join(args.ckpt_path, args.dataset, args.task_info, 'stage1', 'models')
        args.task_info = args.task_info+ '_mtl'
        args.lr = args.lr * 1.0

    return init_args(args)

def init_args(args):
    # create folder for models/outputs/logs of stage1/stage2
    args.root_s1 = os.path.join(args.ckpt_path, args.dataset, args.task_info, 'stage1')
    args.model_path_s1 = os.path.join(args.root_s1, 'models' )
    args.output_path_s1 = os.path.join(args.root_s1, "outputs")
    # args.log_path_s1 = os.path.join(args.root_s1, "logs")
    args.log_path_s1 = os.path.join(args.root_s1, "logs_" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    args.root_s2 = os.path.join(args.ckpt_path, args.dataset, args.task_info, 'stage2')
    args.model_path_s2 = os.path.join(args.root_s2, 'models' )
    args.output_path_s2 = os.path.join(args.root_s2, "outputs")
    # args.log_path_s2 = os.path.join(args.root_s2, "logs")
    args.log_path_s2 = os.path.join(args.root_s2, "logs_"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    for dir in [args.model_path_s1, args.log_path_s1, args.output_path_s1,
                args.model_path_s2, args.log_path_s2, args.output_path_s2]:
        mkdir(dir)

    # mapping parameters of string format
    args.act_thresh_cas = eval(args.act_thresh_cas)
    args.act_thresh_agnostic = eval(args.act_thresh_agnostic)
    args.lambdas = eval(args.lambdas)
    args.tIoU_thresh = eval(args.tIoU_thresh)
    
    # get list of class name 
    args.class_name_lst = _CLASS_NAME[args.dataset]
    args.num_class = len(args.class_name_lst)

    # define format of test information
    if args.cfg == 'thumos' or args.cfg == 'gtea' or args.cfg == 'beoid':
        args.test_info = {
            "step": [], "test_acc": [], 'loss': [], 'elapsed': [], 'now': [],
            "average_mAP[0.1:0.7]": [], "average_mAP[0.1:0.5]": [], "average_mAP[0.3:0.7]": [],
            "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], "mAP@0.4": [],
            "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": [], 'dataset':'thumos'
    }
    if args.cfg == 'activitynet':
        args.test_info = {
                "step": [], "test_acc": [], 'loss': [], 'elapsed': [], 'now': [],
                "average_mAP[0.5:0.95]": [],
                "mAP@0.50": [], "mAP@0.55": [], "mAP@0.60": [], "mAP@0.65": [], "mAP@0.70": [], "mAP@0.75": [], "mAP@0.80": [], "mAP@0.85": [], "mAP@0.90": [], "mAP@0.95": [], 'dataset':'activitynet'
            }

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
