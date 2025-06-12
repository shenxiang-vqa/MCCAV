# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os, json, torch, pickle
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from openvqa.models.model_loader import ModelLoader
from openvqa.datasets.dataset_loader import EvalLoader


# Evaluation
@torch.no_grad()
def test_engine(__C, dataset, state_dict=None, validation=False):

    # Load parameters
    if __C.CKPT_PATH is not None:
        print('Warning: you are now using CKPT_PATH args, '
              'CKPT_VERSION and CKPT_EPOCH will not work')

        path = __C.CKPT_PATH
    else:
        path = __C.CKPTS_PATH + \
               '/ckpt_' + __C.CKPT_VERSION + \
               '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'

    # val_ckpt_flag = False
    if state_dict is None:
        # val_ckpt_flag = True
        print('Loading ckpt from: {}'.format(path))
        state_dict = torch.load(path)['state_dict']
        print('Finish!')

        if __C.N_GPU > 1:
            state_dict = ckpt_proc(state_dict)

    # Store the prediction list
    # qid_list = [ques['question_id'] for ques in dataset.ques_list]
    ans_ix_list = []
    pred_list = []
    lang_sa_list = []
    img_sa_list = []
    img_ga_list = []
    lang_ga_list = []
    lang_alpha_list = []
    img_alpha_list = []

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size
    )
    net.cuda()
    net.eval()

    if __C.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=__C.DEVICES)

    net.load_state_dict(state_dict)

    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM
    )

    for step, (
            frcn_feat_iter,
            grid_feat_iter,
            vit_feat_iter,
            bbox_feat_iter,
            w_feat_iter,
            h_feat_iter,
            spa_graph_iter,
            ques_ix_iter,
            ques_tensor_iter,
            ans_iter
    ) in enumerate(dataloader):

        print("\rEvaluation: [step %4d/%4d]" % (
            step,
            int(data_size / __C.EVAL_BATCH_SIZE),
        ), end='          ')

        frcn_feat_iter = frcn_feat_iter.cuda()
        grid_feat_iter = grid_feat_iter.cuda()
        vit_feat_iter = vit_feat_iter.cuda()
        bbox_feat_iter = bbox_feat_iter.cuda()
        w_feat_iter = w_feat_iter.cuda()
        h_feat_iter = h_feat_iter.cuda()
        spa_graph_iter = spa_graph_iter.cuda()
        ques_ix_iter = ques_ix_iter.cuda()
        ques_tensor_iter = ques_tensor_iter.cuda()

        pred,lang_sa, img_sa, img_ga, lang_ga, lang_alpha, img_alpha = net(
            frcn_feat_iter,
            grid_feat_iter,
            vit_feat_iter,
            bbox_feat_iter,
            w_feat_iter,
            h_feat_iter,
            spa_graph_iter,
            ques_ix_iter,
            ques_tensor_iter
        )
        pred_np = pred.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)
        lang_sa= lang_sa.cpu().data.numpy()
        img_sa = img_sa.cpu().data.numpy()
        img_ga = img_ga.cpu().data.numpy()
        #lang_ga = lang_ga.cpu().data.numpy()
        lang_alpha = lang_alpha.cpu().data.numpy()
        img_alpha = img_alpha.cpu().data.numpy()
        
        
        lang_sa_list.append(lang_sa[:,:,:8,:8])
        img_sa_list.append(img_sa[:,:,:22,:22])
        img_ga_list.append(img_ga[:,:,:22,:8])
        #lang_ga_list.append(lang_ga)
        lang_alpha_list.append(lang_alpha)
        img_alpha_list.append(img_alpha)
        
        # save i-th attntion map
        if lang_sa_list.__len__() == 736:
            np.savez('att_map.npz', lang_sa = lang_sa_list[-1], 
            #lang_ga = lang_ga_list[-1], 
            lang_alpha = lang_alpha_list[-1],
            img_sa = img_sa_list[-1],
            img_ga = img_ga_list[-1],
            img_alpha = img_alpha_list[-1])
        
            '''
            with open("lang_sa.txt", 'w') as f:
                f.write(lang_sa_list[-1] + '\n')

            with open("img_sa.txt", 'w') as f:
                f.write(img_sa_list[-1] + '\n')

            with open("img_ga.txt", 'w') as f:
                f.write(img_ga_list[-1] + '\n')   
                
            with open("lang_ga.txt", 'w') as f:
                f.write(lang_ga_list[-1] + '\n')

            with open("lang_alpha.txt", 'w') as f:
                f.write(lang_alpha_list[-1]  + '\n')

            with open("img_alpha.txt", 'w') as f:
                f.write(img_alpha_list[-1] + '\n')
            '''
            break 
                   

        # Save the answer index
        if pred_argmax.shape[0] != __C.EVAL_BATCH_SIZE:
            pred_argmax = np.pad(
                pred_argmax,
                (0, __C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                mode='constant',
                constant_values=-1
            )

        ans_ix_list.append(pred_argmax)

        # Save the whole prediction vector
        if __C.TEST_SAVE_PRED:
            if pred_np.shape[0] != __C.EVAL_BATCH_SIZE:
                pred_np = np.pad(
                    pred_np,
                    ((0, __C.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                    mode='constant',
                    constant_values=-1
                )

            pred_list.append(pred_np)

    print('')
    ans_ix_list = np.array(ans_ix_list).reshape(-1)


    if validation:
        if __C.RUN_MODE not in ['train']:
            result_eval_file = __C.CACHE_PATH + '/result_run_' + __C.CKPT_VERSION
        else:
            result_eval_file = __C.CACHE_PATH + '/result_run_' + __C.VERSION
    else:
        if __C.CKPT_PATH is not None:
            result_eval_file = __C.RESULT_PATH + '/result_run_' + __C.CKPT_VERSION
        else:
            result_eval_file = __C.RESULT_PATH + '/result_run_' + __C.CKPT_VERSION + '_epoch' + str(__C.CKPT_EPOCH)


    if __C.CKPT_PATH is not None:
        ensemble_file = __C.PRED_PATH + '/result_run_' + __C.CKPT_VERSION + '.pkl'
    else:
        ensemble_file = __C.PRED_PATH + '/result_run_' + __C.CKPT_VERSION + '_epoch' + str(__C.CKPT_EPOCH) + '.pkl'


    if __C.RUN_MODE not in ['train']:
        log_file = __C.LOG_PATH + '/log_run_' + __C.CKPT_VERSION + '.txt'
    else:
        log_file = __C.LOG_PATH + '/log_run_' + __C.VERSION + '.txt'

    EvalLoader(__C).eval(dataset, ans_ix_list, pred_list, result_eval_file, ensemble_file, log_file, validation)


def ckpt_proc(state_dict):
    state_dict_new = {}
    for key in state_dict:
        state_dict_new['module.' + key] = state_dict[key]
        # state_dict.pop(key)

    return state_dict_new
