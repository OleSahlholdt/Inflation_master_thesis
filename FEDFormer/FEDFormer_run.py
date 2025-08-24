import os
import shutil
import torch
from tqdm import tqdm
from exp.exp_main import Exp_Main
from default_args import FEDFormer_default_args
import copy
import optuna
import joblib

Exp = Exp_Main

def objective(trial, args, kernel_sizes, valid_seq_label_pairs):
    args.moving_avg = trial.suggest_categorical("moving_avg", kernel_sizes)
    seq_len, label_len = trial.suggest_categorical("seq_label_pair", valid_seq_label_pairs)
    args.seq_len = seq_len
    args.label_len = label_len

    args.n_heads = trial.suggest_categorical("n_heads", [8, 16])
    args.e_layers = trial.suggest_categorical("e_layers", [2, 3, 4])

    args.version = trial.suggest_categorical("version", ["Fourier", "Wavelets"])

    args.d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    args.d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024, 2048])
    args.dropout = trial.suggest_float("dropout", 0.0, 0.3)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    setting = create_experiment_setting(args)
    exp = Exp(args)
    print(f'>>>>>>> start optuna trial: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>')

    try:
        loss = exp.cross_validate(setting)
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")

    torch.cuda.empty_cache()
    return loss

def run_experiment(args, seq_lengths, kernel_sizes, task_id):
    """
    Runs the experiment for a given task ID and prediction length.
    """
    args.task_id = task_id
    args.do_predict = True

    print(f"Pred length: {args.pred_len}")

    for model in ['FEDformer']:
        args.model = model
        for i in tqdm(range(0, 118)):
            args.idx = i
            month = (i % 12) + 1
            if i == 0 or month == 12:
                best_loss, best_args, best_setting, study = perform_cross_validation(args, seq_lengths, kernel_sizes)
            # Create a new copy for each idx to avoid mutation issues
            current_args = copy.deepcopy(best_args)
            current_args.idx = i
            setting = train_best_model(current_args)
            if args.do_predict:
                perform_prediction(setting, current_args)
            folder_path = './results/' + setting + '/'
            # save after tuning
            joblib.dump(study, folder_path + "optuna_study.pkl")

def perform_cross_validation(args, seq_lengths, kernel_sizes, n_trials=50):
    valid_seq_label_pairs = []
    for seq_len in seq_lengths:
        for label_len in seq_lengths:
            if label_len <= seq_len:
                valid_seq_label_pairs.append((seq_len, label_len))
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, args, kernel_sizes, valid_seq_label_pairs),
        n_trials=n_trials,
    )

    best_loss = study.best_value
    best_params = study.best_params

    best_args = copy.deepcopy(args)
    for k, v in best_params.items():
        setattr(best_args, k, v)

    best_setting = create_experiment_setting(best_args)
    print(f"Best params: {best_params}, Best loss: {best_loss}")

    return best_loss, best_args, best_setting, study

def train_best_model(args):
    exp = Exp(args)
    setting = create_experiment_setting(args)
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)
    return setting

def perform_prediction(setting, args):
    print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp = Exp(args)
    exp.predict(setting, True)
    checkpoint_dir = r"FEDFormer_checkpoints/"
    for ckpt in os.listdir(checkpoint_dir):
        shutil.rmtree(rf'{checkpoint_dir}/{ckpt}')

def create_experiment_setting(args):
    return '{}_{}_seqlen{}_labellen{}_dm{}_ma{}_dff{}_drop{:.2f}_lr{:.0e}_bs{}'.format(
        args.idx,
        args.task_id,
        args.seq_len,
        args.label_len,
        args.d_model,
        args.moving_avg,
        args.d_ff,
        args.dropout,
        args.learning_rate,
        args.batch_size,
    )

args = FEDFormer_default_args()
args.batch_size = 8

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    seq_lengths = [12, 24, 36]
    kernel_sizes = [5, 9, 13, 25]
    horizon = f"h{args.pred_len}"
    run_experiment(args, seq_lengths, kernel_sizes, task_id=f"FEDFormer_{horizon}")