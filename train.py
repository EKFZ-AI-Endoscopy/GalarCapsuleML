import datetime
import os
import random
import time
from pathlib import Path
from sys import stderr

import mlflow
import numpy as np
import questionary
import torch
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import DataSetPreparationExternalSplit

from metrics import (MetricLogger, write_confusion_matrix,
                     write_confusion_matrix_single)
from model import prepare_model
from options import parser

import copy
from utils import ClassificationType, MetricMonitor


def train(data_loader: DataLoader, mode: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.loss._Loss, device: str, epoch: int, features: list, classification_type: ClassificationType, log, metric_logger: MetricLogger):
    
    log('')
    log('----------EPOCH {} {} ----------'.format(epoch, mode))

    metric_monitor = MetricMonitor()

    assert(mode in {"train","val","test"})
    
    if mode == "train":
        model.train()
    elif mode in {"val","test"}:
        model.eval()

    cumulative_loss = 0

    stream = tqdm(data_loader)

    for index, (images, target) in enumerate(stream, start=1):

        optimizer.zero_grad()

        if classification_type == ClassificationType.SECTION_MULTICLASS or classification_type == ClassificationType.MULTICLASS_TECHNICAL or args.dual_output:
            target = torch.argmax(target, dim=1).type(torch.int64)
            outputs = model(images.to(device, non_blocking=True))

            loss = criterion(outputs, target.to(device).long())

        elif classification_type == ClassificationType.MULTILABEL_TECHNICAL or classification_type == ClassificationType.MULTILABEL_PATHLOGICAL:
            outputs = model(images.to(device))
            
            loss = criterion(outputs, target.to(device).float())

            outputs = torch.sigmoid(outputs)
        else:
            outputs = model(images.to(device, non_blocking=True))
            loss = criterion(outputs, target.to(device).float())

            outputs = torch.sigmoid(outputs.detach().cpu().squeeze(1))
            target = target.detach().cpu().squeeze(1)
            

        cumulative_loss += loss.detach().cpu().item() * target.size(0)

        metric_monitor.update("Loss", loss.detach().cpu().item())

        if mode == "train":
            loss.backward()
            optimizer.step()

        # Will be automatically accumulated in metric_logger
        metric_logger.update(copy.deepcopy(outputs.detach().cpu()), copy.deepcopy(target.detach().cpu()))

        stream.set_description("Epoch: {epoch}. {mode}.       {metric_monitor}".format(epoch=epoch, mode=mode, metric_monitor=metric_monitor))
    
    cumulative_loss /= len(data_loader.dataset)

    f1, conf_mat, metric_dict = metric_logger.compute(show_plt = mode in {"val","test"}, fig_name=f'{mode}_ep{epoch}')

    return f1,cumulative_loss,model,optimizer,conf_mat, metric_dict


def logging(mode:str,writer,log,epoch:int,f1:float,loss:float) -> None:

    log(f"Epoche {epoch} {mode}: (loss {loss:.4f}, F1 {f1:.4f})")

    # Writer for tensorboard readout
    if writer:
        writer.add_scalar(f"{mode}/Loss", loss, epoch)
        writer.add_scalar(f"{mode}/F1-Score", f1, epoch)


def main(args) -> None:
    torch.set_printoptions(threshold=10000)
    # Set tracking URI
    MODEL_REGISTRY = Path(args.mlflow_path)
    Path(MODEL_REGISTRY).mkdir(exist_ok=True) # create experiments dir
    mlflow.set_tracking_uri("file:///" + str(MODEL_REGISTRY.absolute()))


    if args.freeze and not args.pretrained:
        if not questionary.confirm("WARNING: You are about to train your network from scratch but most layers are frozen. Do you want to continue?").ask():
            exit()

    if args.debug:
        if not questionary.confirm("WARNING: You are about to start training in debug mode with only a small subset of the dataset. Do you want to continue?").ask():
            exit()
        else:
            print('DEBUG - SMALL DATASET')

    if args.epochs < 10:
        if not questionary.confirm("WARNING: Unrealistically low number of epochs. Do you want to continue anyways?").ask():
            exit()

    if args.weighted_loss and args.weighted_sampling:
        log('ERROR: WEIGHTED SAMPLING AND WEIGHTED LOSS SHOULD NOT BE USED CONCURRENTLY AS IT HAS BEEN IMPLEMENTED RIGHT NOW. Future: Implement weighted loss to "fill gaps" between custom weighted sampling and even distribution.')
        exit()

    if args.static_random_downsampling and (args.weighted_loss or args.weighted_sampling):
        log('ERROR: Do not use static random downsampling concurrently with weighted loss or weighted sampling. This will propably not do what you want!')
        exit()
    
    if args.static_random_downsampling:
        if args.weights == '':
            log('ERROR: Please give class weights if using static random downsampling!')
            exit()

    run_folds = args.folds.split('-')

    print(run_folds)

    assert(len(run_folds) <= args.num_folds)

    for index, fold in enumerate(run_folds):
        assert int(fold) < args.num_folds

    # exclude some with no or close to zero occurrences
    #features_pathological = ['z-line', 'pylorus', 'ampulla of vater', 'ileocecal valve', 'ulcer', 'polyp', 'active bleeding', 'blood', 'erythema', 'erosion' , 'angiectasia' , 'CED', 'foreign body' , 'esophagitis', 'varices', 'hematin', 'celiac','cancer','lymphangioectasis']
    features_pathological = ['z-line', 'pylorus', 'ampulla of vater', 'ileocecal valve', 'ulcer', 'polyp', 'blood', 'erythema', 'erosion' , 'angiectasia']

    features_technical_multiclass = ['good view', 'reduced view', 'no view']
    features_technical_multilabel = ['dirt', 'bubbles']
    features_input = args.training_features.replace('\'', '').replace('"','').split(',')
    features_available = ['bubbles', 'dirt', 'reduced view', 'no view', 'esophagus', 'stomach', 'small intestine', 'mouth', 'colon', 'z-line', 'pylorus', 'ampulla of vater', 'ileocecal valve', 'ulcer', 'polyp', 'active bleeding', 'blood', 'erythema', 'erosion' , 'angiectasia' , 'CED', 'foreign body' , 'esophagitis', 'varices', 'hematin', 'celiac','cancer','lymphangioectasis']

    classification_type = ClassificationType.SINGLELABEL

    if len(features_input) > 1 or len(features_input) < 1:
        print('At the moment only one feature for training is supported.')
        exit()

    print('features_input[0]: {}'.format(features_input[0]))
    if features_input[0] == 'section':
        classification_type = ClassificationType.SECTION_MULTICLASS
    elif features_input[0] == 'all':
        classification_type = ClassificationType.MULTILABEL
    elif features_input[0] == 'technical_multiclass':
        classification_type = ClassificationType.MULTICLASS_TECHNICAL
    elif features_input[0] == 'technical_multilabel':
        classification_type = ClassificationType.MULTILABEL_TECHNICAL
    elif features_input[0] == 'pathological':
        classification_type = ClassificationType.MULTILABEL_PATHLOGICAL

    if not classification_type == ClassificationType.SINGLELABEL and args.dual_output:
        print('ERROR: Dual output is not supported for multilabel classification.')
        exit()

    if features_input[0] not in features_available and not classification_type == ClassificationType.SECTION_MULTICLASS and not classification_type == ClassificationType.MULTILABEL and not classification_type == ClassificationType.MULTILABEL_TECHNICAL and not classification_type == ClassificationType.MULTILABEL_PATHLOGICAL and not classification_type == ClassificationType.MULTICLASS_TECHNICAL:
       
        breakpoint()
       
        print('Features not supported')
        exit()
    
    technical_annotation = False

    if classification_type == ClassificationType.SECTION_MULTICLASS:
        features = ['mouth', 'esophagus', 'stomach', 'small intestine', 'colon']
    elif classification_type == ClassificationType.MULTILABEL:
        features = features_available
    elif classification_type == ClassificationType.MULTILABEL_TECHNICAL:
        features = features_technical_multilabel
        technical_annotation = True
    elif classification_type == ClassificationType.MULTICLASS_TECHNICAL:
        features = features_technical_multiclass
        technical_annotation = True
    elif classification_type == ClassificationType.MULTILABEL_PATHLOGICAL:
        features = features_pathological
    else:
        features = features_input
    
    # technical_annotation = features[0] in features_technical

    if args.weights != '':
        weights = args.weights.split(',')

        if len(weights) != len(features):
            log(f'The number of given weights is not equal to the numbers of features: {len(weights)} != {len(features)}')
            exit()

        for weight in weights:
            if float(weight) > 1:
                log(f'WARNING: WEIGHT {i}={weight} > 1!')
                exit()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## make output directory
    date_=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    features_name = features[:10]

    output_path = os.path.join(args.output_dir,f"{date_}_{args.run_prefix}{'-'.join(features_name)}")
    os.makedirs(output_path)

    ## logging in text_file
    log_file = open(os.path.join(output_path,"log.txt"), "a")
    def log(msg):
        print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
        log_file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)
        log_file.flush()
        os.fsync(log_file)
    
    if technical_annotation:
        log('USE TECHNICAL VIDEOS')
    if classification_type == ClassificationType.SECTION_MULTICLASS:
        log('USE ANATOMICAL (Section) LABELS')
    elif classification_type == ClassificationType.MULTILABEL_TECHNICAL:
        log('USE TECHNICAL MULTILABEL LABELS')
    elif classification_type == ClassificationType.MULTICLASS_TECHNICAL:
        log('USE TECHNICAL MULTICLASS LABELS')    
    elif classification_type == ClassificationType.MULTILABEL_PATHLOGICAL:
        log('USE PATHOLOGICAL LABELS')

    log('USE {}-k fold'.format(args.num_folds))
    log('Running ML for the following folds: {}'.format(run_folds))

    log('Output directory: {}'.format(output_path))

    log("With parameters: ")
    for arg in sorted(vars(args)):
        log("\t" + str(arg) + " : " + str(getattr(args, arg)))
    
    log("Training with features: {}".format(features))

    args_dict = {}
    for arg in vars(args):
        args_dict[str(arg)] = getattr(args, arg)
        
    ## tensorboard writer 
    writer = SummaryWriter(log_dir=output_path)
    writer.add_text("Args", str(args_dict), global_step=0)

    label_name_dict = {}
    for k,v in enumerate(features):
        label_name_dict[k] = v

    SEED=1337
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if args.pretrained:
        log('START FINE-TUNING')
    else:
        log('START LEARNING FROM SCRATCH')

    for fold in run_folds:
        fold = int(fold)
        log('')
        log('{} FOLD {} {}'.format(10*'-',fold, 10*'-'))
        mlflow.set_experiment(experiment_name="experiment-{}-{}k-fold-{}".format(features_input[0],args.num_folds,fold))

        output_path_fold = os.path.join(output_path, "fold{}".format(fold))
        os.makedirs(output_path_fold)

        datasetprep = DataSetPreparationExternalSplit(classification_type, log, fold=fold, args=args)
        train_loader, val_loader, test_loader = datasetprep.prepare_dataset(args, features)

        # always load fresh model for each fold!
        model, optimizer, scheduler = prepare_model(args, log, features, classification_type, device)

        ## Loss function
        if classification_type == ClassificationType.SECTION_MULTICLASS or classification_type == ClassificationType.MULTICLASS_TECHNICAL or args.dual_output:
            criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(datasetprep.train_weight).cuda() if args.weighted_loss else None)

        elif classification_type == ClassificationType.MULTILABEL_TECHNICAL or classification_type == ClassificationType.MULTILABEL_PATHLOGICAL:
            criterion = torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor(datasetprep.train_weight).cuda() if args.weighted_loss else None)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

        log('TRAINING DATASET: {} images'.format(len(train_loader)))
        log('VALIDATION DATASET: {} images'.format(len(val_loader)))

        best_f1 = 0

        confusion_matrices = []

        with mlflow.start_run(run_name="run-{}-{}-{}-{}-pretrained{}-freeze{}-batch_size{}".format(args.epochs, args.learning_rate, args.optimizer, args.model, args.pretrained, args.freeze, args.batch_size)):

            early_stopping = EarlyStopping(depth=5, ignore=10, method="consistency")

            for epoch in range(1,args.epochs+1):

                ## training
                metric_logger = MetricLogger(classification_type, log, features, fig_save_path=output_path_fold)
                f1_train,loss_train,model,optimizer,conf_mat,metric_dict = train(train_loader,"train",model,optimizer,criterion,device,epoch,features,classification_type,log, metric_logger)
                log('')
                logging("train",writer,log,epoch,f1_train,loss_train)

                with torch.no_grad():
                    ##Validation
                    metric_logger = MetricLogger(classification_type, log, features, fig_save_path=output_path_fold)
                    f1_val,loss_val,model,optimizer,conf_mat,metric_dict = train(val_loader,"val",model,optimizer,criterion,device,epoch,features,classification_type,log, metric_logger)
                    confusion_matrices.append(conf_mat)
                    log('')
                    logging("val",writer,log,epoch,f1_val,loss_val)

                    mlflow.log_metrics({"train_loss": loss_train, "val_loss": loss_val, "train_f1": f1_train, "val_f1": f1_val}, step=epoch)
                
                scheduler.step(loss_val)

                ## save Checkpoint
                if epoch % 5 == 0 or epoch == args.epochs:
                    current_state = {'epoch': epoch,
                                        'model_weights': model.state_dict(),
                                        'optimizer': optimizer.state_dict(),
                                        'args': args_dict,
                                        }

                    model_path = os.path.join(output_path_fold,f"checkpoint.pth.tar")
                    torch.save(current_state,model_path)
                    log(f"Saved checkpoint to: {model_path}")

                ### save best model on validation set
                if epoch  > 0 and f1_val > best_f1:
                    
                    best_f1 = f1_val

                    current_state = {'epoch': epoch,
                                        'model_weights': model.state_dict(),
                                        'optimizer': optimizer.state_dict(),
                                        'args': args_dict,
                                        'f1_val': f1_val,
                                        }
                    model_path = os.path.join(output_path_fold,f"model_best_f1.pth.tar")
                    torch.save(current_state,model_path)
                    log(f"Saved Model with best Validation F1 to: {model_path}")
                
                if early_stopping.check(loss_val):
                    log(f'EARLY STOPPING at epoch {epoch}')
                    break
                
            ## test best model:
            if args.epochs >= 10:
                log("Testing best Validation Model")
                with torch.no_grad():
                    model.load_state_dict(torch.load(os.path.join(output_path_fold,f"model_best_f1.pth.tar"))['model_weights'])
                    metric_logger = MetricLogger(classification_type, log, features, fig_save_path=output_path_fold)
                    f1_test,loss_test,model,optimizer,conf_mat,metric_dict = train(test_loader,"test",model,optimizer,criterion,device,epoch,features,classification_type,log, metric_logger)
                    log('')
                    log(f"Test: (loss {loss_test:.4f}, F1 {f1_test:.4f})")
                    if classification_type == ClassificationType.MULTILABEL_TECHNICAL or classification_type == ClassificationType.MULTILABEL_PATHLOGICAL:
                        for feature, conf_mat_s in zip(features, conf_mat):
                            write_confusion_matrix_single(conf_mat_s, feature, epoch+1, output_path_fold)
                    else:
                        write_confusion_matrix(conf_mat, features, 0, output_path_fold)

                    mlflow.log_metrics({"f1": f1_test})
                    mlflow.log_metrics(metric_dict)

            mlflow.log_params(vars(args))            

    log_file.close()
    writer.flush()
    writer.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
