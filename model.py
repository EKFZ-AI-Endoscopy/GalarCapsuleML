from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights, vit_l_32, ViT_L_32_Weights, ViT_B_16_Weights, vit_b_16, efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights
from utils import ClassificationType
import torch
import timm


def prepare_model(args, log, features: list, classification_type: ClassificationType, device: str) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Get model, optimizer and learning rate scheduler according to the parameters. """

    number_features = len(features)

    if args.dual_output:
        number_features = 2

    if classification_type == ClassificationType.SECTION_MULTICLASS:
        number_features = 5

    if args.model == 'resnet50':
        model = resnet50(weights="IMAGENET1K_V2" if args.pretrained else None)
        last_layer_in_features = 2048
    elif args.model == 'resnet34':
        model = resnet34(weights=ResNet34_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 512
    elif args.model == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 512
    elif args.model == 'resnet101':
        model = resnet101(weights=ResNet101_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 2048
    elif args.model == 'resnet152':
        model = resnet152(weights=ResNet152_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 2048
    elif args.model == 'vit_l_32':
        model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 1024
    elif args.model == 'vit_b_16':
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 768
    elif args.model == 'efficientnet_v2_m':
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 1280
    elif args.model == 'efficientnet_v2_s':
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 1280
    elif args.model == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 1280
    elif args.model == 'efficientnet_b1':
        model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 1280
    elif args.model == 'efficientnet_b2':
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT if args.pretrained else None)
        last_layer_in_features = 1408
    elif args.model == 'inceptionresnetv2':
        model = timm.create_model('inception_resnet_v2', pretrained=True if args.pretrained else None, num_classes=number_features)

    else:
        log('Model is not yet supported. Please add to model.py!')
        exit()
    
    FREEZE = args.freeze != 0

    if FREEZE:
        log('FREEZING LAYERS')
        for param in model.parameters():
            param.requires_grad = False

    if 'vit' in args.model:
        model.heads.head = torch.nn.Linear(in_features=last_layer_in_features, out_features=number_features, bias=True)
        for param in model.heads.parameters():
            param.requires_grad = True
    elif 'efficientnet' in args.model:
        model.classifier[1] = torch.nn.Linear(in_features=last_layer_in_features, out_features=number_features, bias=True)
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif 'inceptionresnet' in args.model:
        model.classif = torch.nn.Sequential(
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(in_features=1536, out_features=number_features, bias=True)
        )

    elif 'resnet' in args.model:
        # model.fc = torch.nn.Linear(in_features=last_layer_in_features, out_features=number_features, bias=True)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(in_features=last_layer_in_features, out_features=number_features, bias=True)
        )
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        print('MODEL MIGHT BE INCOMPATIBLE')
    model.to(device)

    log('Number of total parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    log('Number of trainable parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if args.optimizer == 'adam':
        log('Use Adam optimizer')
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adamW':
        log('Use AdamW optimizer')
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    else:
        log('Use SGD optimizer (take note that you might have to adjust the learning rate!)')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    return model, optimizer, scheduler
