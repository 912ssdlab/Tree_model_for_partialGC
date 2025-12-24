Status blk_Inqueue(struct ssd_info *ssd, int channel, int chip, int die, int plane, int *blk_id){
    int i;
    int block = -1;
    int active_block = 0;
    unsigned int invalid_page = 0;
    unsigned int superblock_invalid_page_num = 0;

    PyObject *sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_FromString("."));
    PyObject *pModule = PyImport_ImportModule("DS_warm");
    if (pModule == NULL) {
        PyErr_Print();
        printf("can't import Python module\n");
        return ERROR;
    }
    PyObject *pFunc = PyObject_GetAttrString(pModule, "train_and_evaluate");
    if (pFunc == NULL || !PyCallable_Check(pFunc)) {
        PyErr_Print();
        printf("can't get Python func\n");
        return ERROR;
    }

    if(find_active_block(ssd,channel,chip,die,plane)!=SUCCESS)
    {
        printf("\n\n Error in uninterrupt_gc().\n");
        return ERROR;
    }
    active_block=ssd->channel_head[channel].chip_head[chip].die_head[die].plane_head[plane].active_block;
    int start_superblock = (chip == 0) ? 0 : ssd->parameter->block_plane;
    int end_superblock = (chip == 0) ? ssd->parameter->block_plane : ssd->parameter->block_plane * ssd->parameter->chip_channel[0];
    for(i=start_superblock;i<end_superblock;i++) {
        int flag = 0;
        for (int j = 0; j < ssd->parameter->channel_number; j++) {
            if (ssd->channel_head[j].chip_head[chip].die_head[die].plane_head[plane].blk_head[ssd->superblock[i].super_blk_loc[j].blk].free_page_num >
                0
                ||
                ssd->channel_head[j].chip_head[chip].die_head[die].plane_head[plane].blk_head[ssd->superblock[i].super_blk_loc[j].blk].super_flag ==
                1) {
                flag = 1;
                break;
            }
            if (j == channel) {
                continue;
            }
        }
        if (flag == 1) {
            continue;
        }
        for (int ch = 0; ch < ssd->parameter->channel_number; ch++) {
            superblock_invalid_page_num += ssd->channel_head[ch].chip_head[chip].die_head[die].plane_head[plane].blk_head[ssd->superblock[i].super_blk_loc[ch].blk].invalid_page_num;
        }
        if ((active_block != i) && (superblock_invalid_page_num > invalid_page) &&
            (ssd->superblock[i].is_softSB_inQue == 0))
        {
            int SB_chip = i /ssd->parameter->block_plane;
            int SB_block = i %ssd->parameter->block_plane;
           ssd->superblock[i].last_update_time = ssd->current_time - ssd->superblock[i].update_time;    
            for(int ch = 0; ch< ssd->parameter->channel_number; ch++){
                ssd->superblock[i].invalid_page_count += ssd->channel_head[ch].chip_head[SB_chip].die_head[0].plane_head[0].blk_head[SB_block].invalid_page_num;
            }
            float diffB;
            for(int ch = 0; ch< ssd->parameter->channel_number; ch++){
                float Avg = ssd->superblock[i].invalid_page_count/8;
                diffB += (Avg-ssd->channel_head[ch].chip_head[SB_chip].die_head[die].plane_head[plane].blk_head[SB_block].invalid_page_num)*(Avg-ssd->channel_head[ch].chip_head[SB_chip].die_head[die].plane_head[plane].blk_head[SB_block].invalid_page_num);
            }
            ssd->superblock[i].Invaild_variance = diffB/8;

            // use model
            PyObject *pArgs = Py_BuildValue("(KIfKi)", ssd->superblock[i].last_update_time, ssd->superblock[i].invalid_page_count, ssd->superblock[i].Invaild_variance, ssd->superblock[i].Avg_update_time, ssd->superblock[i].neighbor_update);  // 空参数元组
            PyObject *pResult = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pResult == NULL) {
                PyErr_Print();
                Py_XDECREF(pFunc);
                Py_XDECREF(pModule);
                return ERROR;
            }
            int pred = 0;
            if (PyArg_ParseTuple(pResult, "i", &pred)) {
                if(pred == 0){
                    invalid_page = superblock_invalid_page_num;
                    block = i;
                }
            } else {
                PyErr_Print();
                printf("false\n");
            }
            Py_XDECREF(pResult);
        }
        superblock_invalid_page_num = 0;
    }
    *blk_id = block;
    if (block == -1)
    {
        return ERROR;
    }else{
        return SUCCESS;
    }