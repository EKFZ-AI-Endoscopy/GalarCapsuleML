from utils import ClassificationType
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import torch
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassSpecificity, MulticlassAUROC, MulticlassMatthewsCorrCoef, MulticlassConfusionMatrix, MulticlassROC
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, MultilabelPrecision, MultilabelRecall, MultilabelSpecificity, MultilabelAUROC, MultilabelMatthewsCorrCoef, MultilabelRankingAveragePrecision, MultilabelConfusionMatrix, MultilabelROC
from torchmetrics.classification import BinaryPrecision, BinaryAccuracy, BinaryAUROC, BinaryConfusionMatrix, BinaryF1Score, BinaryRecall, BinarySpecificity, BinaryROC
import seaborn as sns
import os

def write_confusion_matrix_single(conf_mat:np.ndarray, features:list, epoch:int, output_path:str) -> None:
    """Generate image of confusion matrix for single label / binary classification."""

    figure(figsize = (6,5), dpi=200)

    ax = sns.heatmap(conf_mat, annot=True, xticklabels=['yes', 'no'], yticklabels=['yes', 'no'], cmap='Blues', fmt='.5g')
    ax.set(xlabel="predicitions", ylabel="ground_truth")

    import matplotlib
    matplotlib.use('Agg')
    plt.title('Epoch {} Validation Confusion Matrix'.format(epoch))
    plt.savefig(os.path.join(output_path,f"epochval"+'{}_{}.jpg'.format(epoch, features)), dpi=300, bbox_inches="tight")
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

def write_confusion_matrix(conf_mat:np.ndarray, features:list, epoch:int, output_path:str) -> None:
    """Generate image of confusion matrix for multi label / multi class classification."""
    figure(figsize = (6,5), dpi=200)

    ax = sns.heatmap(conf_mat, annot=True, xticklabels=features, yticklabels=features, cmap='Blues', fmt='.5g')
    ax.set(xlabel="predicitions", ylabel="ground_truth")

    # should fix "Fail to create pixmap with Tk_GetPixmap in TkImgPhotoInstanceSetSize"
    import matplotlib
    matplotlib.use('Agg')
    plt.title('Epoch {} Validation Confusion Matrix'.format(epoch))
    plt.savefig(os.path.join(output_path,f"epochval"+'{}.jpg'.format(epoch)), dpi=300, bbox_inches="tight")
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

class MetricLogger:
    """Automatically accumulates and calculates many metrics according to the classification problem type."""

    def __init__(self, classification_type:ClassificationType, log, features:list, fig_save_path:str) -> None:
        self.classification_type = classification_type
        self.log = log
        self.features = features
        self.fig_save_path = fig_save_path
        self.reset()

    def reset(self):
        """Reset metrics (for each epoch)"""
        if self.classification_type == ClassificationType.SECTION_MULTICLASS or self.classification_type == ClassificationType.MULTICLASS_TECHNICAL:
            self.metric_f1_none = MulticlassF1Score(average='none', num_classes=len(self.features))
            self.metric_f1_micro = MulticlassF1Score(average='micro', num_classes=len(self.features))
            self.metric_f1_macro = MulticlassF1Score(average='macro', num_classes=len(self.features))
            self.metric_acc_none = MulticlassAccuracy(average='none', num_classes=len(self.features))
            self.metric_acc_micro = MulticlassAccuracy(average='micro', num_classes=len(self.features))
            self.metric_acc_macro = MulticlassAccuracy(average='macro', num_classes=len(self.features))
            self.metric_prec_none = MulticlassPrecision(average='none', num_classes=len(self.features))
            self.metric_prec_micro = MulticlassPrecision(average='micro', num_classes=len(self.features))
            self.metric_prec_macro = MulticlassPrecision(average='macro', num_classes=len(self.features))
            self.metric_rec_none = MulticlassRecall(average='none', num_classes=len(self.features))
            self.metric_rec_micro = MulticlassRecall(average='micro', num_classes=len(self.features))
            self.metric_rec_macro = MulticlassRecall(average='macro', num_classes=len(self.features))
            self.metric_spec_none = MulticlassSpecificity(average='none', num_classes=len(self.features))
            self.metric_spec_macro = MulticlassSpecificity(average='macro', num_classes=len(self.features))
            self.metric_spec_micro = MulticlassSpecificity(average='micro', num_classes=len(self.features))
            self.metric_auroc_none = MulticlassAUROC(average='none', num_classes=len(self.features))
            self.metric_auroc_macro = MulticlassAUROC(average='macro', num_classes=len(self.features))
            self.metric_matth_micro = MulticlassMatthewsCorrCoef(num_classes=len(self.features))
            self.metric_roc = MulticlassROC(num_classes=len(self.features))
            self.confusion_matrix = MulticlassConfusionMatrix(num_classes=len(self.features))

        elif self.classification_type == ClassificationType.MULTILABEL_TECHNICAL or self.classification_type == ClassificationType.MULTILABEL_PATHLOGICAL:
            self.metric_f1_none = MultilabelF1Score(average='none', num_labels=len(self.features))
            self.metric_f1_micro = MultilabelF1Score(average='micro', num_labels=len(self.features))
            self.metric_f1_macro = MultilabelF1Score(average='macro', num_labels=len(self.features))
            self.metric_acc_none = MultilabelAccuracy(average='none', num_labels=len(self.features))
            self.metric_acc_micro = MultilabelAccuracy(average='micro', num_labels=len(self.features))
            self.metric_acc_macro = MultilabelAccuracy(average='macro', num_labels=len(self.features))
            self.metric_prec_none = MultilabelPrecision(average='none', num_labels=len(self.features))
            self.metric_prec_micro = MultilabelPrecision(average='micro', num_labels=len(self.features))
            self.metric_prec_macro = MultilabelPrecision(average='macro', num_labels=len(self.features))
            self.metric_rec_none = MultilabelRecall(average='none', num_labels=len(self.features))
            self.metric_rec_micro = MultilabelRecall(average='micro', num_labels=len(self.features))
            self.metric_rec_macro = MultilabelRecall(average='macro', num_labels=len(self.features))
            self.metric_spec_none = MultilabelSpecificity(average='none', num_labels=len(self.features))
            self.metric_spec_macro = MultilabelSpecificity(average='macro', num_labels=len(self.features))
            self.metric_spec_micro = MultilabelSpecificity(average='micro', num_labels=len(self.features))
            self.metric_auroc_none = MultilabelAUROC(average='none', num_labels=len(self.features))
            self.metric_auroc_micro = MultilabelAUROC(average='micro', num_labels=len(self.features))
            self.metric_auroc_macro = MultilabelAUROC(average='macro', num_labels=len(self.features))
            self.metric_matth_micro = MultilabelMatthewsCorrCoef(num_labels=len(self.features))
            self.metric_mlranking = MultilabelRankingAveragePrecision(num_labels=len(self.features))
            self.metric_roc = MultilabelROC(num_labels=len(self.features))
            self.confusion_matrix = MultilabelConfusionMatrix(num_labels=len(self.features))

        else:
            self.metric_f1 = BinaryF1Score()
            self.metric_acc = BinaryAccuracy()
            self.metric_prec = BinaryPrecision()
            self.metric_rec = BinaryRecall()
            self.metric_spec = BinarySpecificity()
            self.metric_auroc = BinaryAUROC()
            self.confusion_matrix = BinaryConfusionMatrix()
            self.metric_roc = BinaryROC()

    def update(self, y_score_props:torch.tensor, y_true:torch.tensor) -> None:
        """Update metrics / values will be saved and accumulated."""

        if self.classification_type == ClassificationType.MULTILABEL_TECHNICAL:
            y_score = y_score_props
        else:
            y_score = y_score_props
            if y_true.dim() == 2:
                y_true = torch.argmax(y_true, dim=1)

        if self.classification_type == ClassificationType.SINGLELABEL:
            if y_score.dim() == 2:
                y_score = torch.argmax(y_score, dim=1)

        try:
            self.confusion_matrix.update(y_score, y_true)
        except Exception as error:
            print(f"Confusion Matrix error: {error}")
            breakpoint()

        if self.classification_type == ClassificationType.SECTION_MULTICLASS or self.classification_type == ClassificationType.MULTILABEL_TECHNICAL or self.classification_type == ClassificationType.MULTILABEL_PATHLOGICAL or self.classification_type == ClassificationType.MULTICLASS_TECHNICAL:
            self.metric_f1_none.update(y_score, y_true)
            self.metric_f1_micro.update(y_score, y_true)
            self.metric_f1_macro.update(y_score, y_true)
            self.metric_acc_none.update(y_score, y_true)
            self.metric_acc_micro.update(y_score, y_true)
            self.metric_acc_macro.update(y_score, y_true)
            self.metric_prec_none.update(y_score, y_true)
            self.metric_prec_micro.update(y_score, y_true)
            self.metric_prec_macro.update(y_score, y_true)
            self.metric_rec_none.update(y_score, y_true)
            self.metric_rec_micro.update(y_score, y_true)
            self.metric_rec_macro.update(y_score, y_true)
            self.metric_spec_none.update(y_score, y_true)
            self.metric_spec_macro.update(y_score, y_true)
            self.metric_spec_micro.update(y_score, y_true)
            self.metric_auroc_none.update(y_score_props, y_true)
            self.metric_auroc_macro.update(y_score_props, y_true)
            self.metric_matth_micro.update(y_score, y_true)
            self.metric_roc.update(y_score_props, y_true)
        else:
            self.metric_f1.update(y_score, y_true)
            self.metric_acc.update(y_score, y_true)
            self.metric_prec.update(y_score, y_true)
            self.metric_rec.update(y_score, y_true)
            self.metric_spec.update(y_score, y_true)
            self.metric_auroc.update(y_score_props, y_true)
            if y_score_props.dim() == 2:
                y_true = torch.nn.functional.one_hot(y_true, num_classes=2)
            self.metric_roc.update(y_score_props, y_true)

        if self.classification_type == ClassificationType.MULTILABEL_TECHNICAL or self.classification_type == ClassificationType.MULTILABEL_PATHLOGICAL:
            self.metric_auroc_micro.update(y_score_props, y_true)


    def compute(self, show_plt:bool=True, fig_name:str="") -> tuple[float, torch.tensor, dict[str, float]]:
        """Calculate scores / metrics using previously accumulated values."""

        conf_mat = self.confusion_matrix.compute()

        metric_dict = {}

        if self.classification_type == ClassificationType.SECTION_MULTICLASS or self.classification_type == ClassificationType.MULTICLASS_TECHNICAL:
            self.log('MULTICLASS SCORES: ')
        elif self.classification_type == ClassificationType.MULTILABEL_TECHNICAL or self.classification_type == ClassificationType.MULTILABEL_PATHLOGICAL:
            self.log('MULTILABEL SCORES: ')
        else:
            self.log('BINARY SCORES: ')
        
        if self.classification_type == ClassificationType.SECTION_MULTICLASS or self.classification_type == ClassificationType.MULTILABEL_TECHNICAL or self.classification_type == ClassificationType.MULTILABEL_PATHLOGICAL or self.classification_type == ClassificationType.MULTICLASS_TECHNICAL:
            self.log('f1-score - classwise: {} / micro: {} / macro: {}'.format(self.metric_f1_none.compute(), self.metric_f1_micro.compute(), self.metric_f1_macro.compute()))
            self.log('accuracy - classwise: {} / micro: {} / macro: {}'.format(self.metric_acc_none.compute(), self.metric_acc_micro.compute(), self.metric_acc_macro.compute()))
            self.log('precision - classwise: {} / micro: {} / macro: {}'.format(self.metric_prec_none.compute(), self.metric_prec_micro.compute(), self.metric_prec_macro.compute()))
            self.log('recall - classwise: {} / micro: {} / macro: {}'.format(self.metric_rec_none.compute(), self.metric_rec_micro.compute(), self.metric_rec_macro.compute()))
            self.log('specificity - classwise: {} / weighted: not implemented / macro: {}'.format(self.metric_spec_none.compute(), self.metric_spec_macro.compute()))
            self.log('Matthews correlation coefficient: {}'.format(self.metric_matth_micro.compute()))
            
            if self.classification_type == ClassificationType.SECTION_MULTICLASS or self.classification_type == ClassificationType.MULTICLASS_TECHNICAL:
                self.log('AUROC - classwise: {} / micro: --- / macro: {}'.format(self.metric_auroc_none.compute(), self.metric_auroc_macro.compute()))
            else:
                self.log('AUROC - classwise: {} / micro: {} / macro: {}'.format(self.metric_auroc_none.compute(), self.metric_auroc_micro.compute(), self.metric_auroc_macro.compute()))
                self.log('Multi Label Rankin Average Precision : {}'.format(self.metric_mlranking.compute()))

            metric_dict['f1_macro'] = self.metric_f1_macro.compute().item()
            metric_dict['f1_micro'] = self.metric_f1_micro.compute().item()
            for i, value in enumerate(self.metric_f1_none.compute()):
                metric_dict['f1_{}'.format(self.features[i])] = value.item()

            metric_dict['accuracy_macro'] = self.metric_acc_macro.compute().item()
            metric_dict['accuracy_micro'] = self.metric_acc_micro.compute().item()
            for i, value in enumerate(self.metric_acc_none.compute()):
                metric_dict['accuracy_{}'.format(self.features[i])] = value.item()

            metric_dict['precision_macro'] = self.metric_prec_macro.compute().item()
            metric_dict['precision_micro'] = self.metric_prec_micro.compute().item()
            for i, value in enumerate(self.metric_prec_none.compute()):
                metric_dict['precision_{}'.format(self.features[i])] = value.item()

            metric_dict['recall_macro'] = self.metric_rec_macro.compute().item()
            metric_dict['recall_micro'] = self.metric_rec_micro.compute().item()
            for i, value in enumerate(self.metric_rec_none.compute()):
                metric_dict['recall_{}'.format(self.features[i])] = value.item()

            metric_dict['specificity_macro'] = self.metric_spec_macro.compute().item()
            metric_dict['specificity_micro'] = self.metric_spec_micro.compute().item()
            for i, value in enumerate(self.metric_spec_none.compute()):
                metric_dict['specificity_{}'.format(self.features[i])] = value.item()

            metric_dict['auroc_macro'] = self.metric_auroc_macro.compute().item()
            if self.classification_type != ClassificationType.SECTION_MULTICLASS and self.classification_type != ClassificationType.MULTICLASS_TECHNICAL:
                metric_dict['auroc_micro'] = self.metric_auroc_micro.compute().item()
            for i, value in enumerate(self.metric_auroc_none.compute()):
                metric_dict['auroc_{}'.format(self.features[i])] = value.item()

        else:

            self.log('f1-score: {}'.format(self.metric_f1.compute()))
            self.log('accuracy: {}'.format(self.metric_acc.compute()))
            self.log('precision: {}'.format(self.metric_prec.compute()))
            self.log('recall: {}'.format(self.metric_rec.compute()))
            self.log('specificity: {}'.format(self.metric_spec.compute()))
            self.log('AUROC: {}'.format(self.metric_auroc.compute()))
            self.log('')

            metric_dict['f1_{}'.format(self.features[0])] = self.metric_f1.compute().item()
            metric_dict['f1_macro'] = self.metric_f1.compute().item()
            metric_dict['accuracy_{}'.format(self.features[0])] = self.metric_acc.compute().item()
            metric_dict['precision_{}'.format(self.features[0])] = self.metric_prec.compute().item()
            metric_dict['recall_{}'.format(self.features[0])] = self.metric_rec.compute().item()
            metric_dict['specificity_{}'.format(self.features[0])] = self.metric_spec.compute().item()
            metric_dict['auroc_{}'.format(self.features[0])] = self.metric_auroc.compute().item()

        fpr, tpr, _ = self.metric_roc.compute()

        if show_plt:
            self.metric_roc.plot()
            plt.savefig(os.path.join(self.fig_save_path, f'roc_{fig_name}.png'), dpi=150)

        return metric_dict['f1_macro'], conf_mat, metric_dict
