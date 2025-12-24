struct ssd_info *get_ppn(struct ssd_info *ssd,unsigned int channel,unsigned int chip,unsigned int die,unsigned int plane,struct sub_request *sub){
    if(ssd->dram->map->map_entry[lpn].state==0)           /*this is the first logical page*/
    {
        if(ssd->dram->map->map_entry[lpn].pn!=0)
        {
            printf("Error in get_ppn()\n");
        }
        ssd->dram->map->map_entry[lpn].pn=find_ppn(ssd,channel,chip,die,plane,block,page);  
        ssd->dram->map->map_entry[lpn].state=sub->state;
        ssd->channel_head[channel].chip_head[chip].die_head[die].plane_head[plane].write_req++;

    }
    else      
    {
        if(ssd->dram->map->map_entry[lpn].is_pg == 1){
            ssd->pagemove_overhead++;
            ssd->dram->map->map_entry[lpn].is_pg = 0;
        }
        ssd->write_req_True++;
        ppn=ssd->dram->map->map_entry[lpn].pn;
        location=find_location(ssd,ppn);
        if(	ssd->channel_head[location->channel].chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].page_head[location->page].lpn!=lpn)
        {
            printf("\nError in get_ppn()\n");
        }
        ssd->channel_head[location->channel].chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].page_head[location->page].valid_state=0;          //表示其中某一页失效，同时标记valid和free状态都为0
        ssd->channel_head[location->channel].chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].page_head[location->page].free_state=0;         //表示某一页失效，同时标记valid和free状态都为0
        ssd->channel_head[location->channel].chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].page_head[location->page].lpn=0;   //删除该页的映射
        ssd->channel_head[location->channel].chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].invalid_page_num++;   //无效页++
        ssd->superblock[location->block+(location->chip * (ssd->parameter->block_plane))].update_count++;

        if(ssd->data_window != (int)(ssd->parameter->block_plane *
                                     ssd->parameter->page_block *
                                     (ssd->parameter->gc_threshold - ssd->parameter->gc_hard_threshold))){   
            int SB_num = location->block + (location->chip*ssd->parameter->block_plane);
            ssd->superblock[SB_num].Avg_update_time = ((ssd->superblock[SB_num].Avg_update_time * ssd->superblock[SB_num].window_udpate_count)+(ssd->current_time - ssd->superblock[SB_num].update_time)) / (ssd->superblock[SB_num].window_udpate_count+1);
            ssd->superblock[SB_num].window_udpate_count++;
            ssd->superblock[SB_num].update_time = ssd->current_time;
            ssd->is_update_flag = 1;
        }else{
            int SB_num = location->block + (location->chip*ssd->parameter->block_plane);
            ssd->superblock[SB_num].window_udpate_count = 0;    
            ssd->superblock[SB_num].Avg_update_time = 0;
            ssd->superblock[SB_num].invalid_page_count = 0;
        }

        if(LRU_Insert(ssd, location->chip, location->block) == FALSE){
            printf("ERROR in LRU Insert. chip is %d, block is %d", location->chip, location->block);
        }

        if((location->page)%3==0){
            ssd->channel_head[location->channel].chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].invalid_lsb_num++;
        }

        free(location);
        location=NULL;
        ssd->dram->map->map_entry[lpn].pn=find_ppn(ssd,channel,chip,die,plane,block,page);
        ssd->dram->map->map_entry[lpn].state=(ssd->dram->map->map_entry[lpn].state|sub->state);
    }

    // collect features
    if(ssd->data_window != 0){
        ssd->data_window--;
    }else{
        if(ssd->is_update_flag != 0 ){   
            int last_SB_update = 0;
            for(int SB = 0; SB < ssd->parameter->block_plane * ssd->parameter->chip_channel[0]; SB++){
                int SB_chip = SB /ssd->parameter->block_plane;
                int SB_block = SB %ssd->parameter->block_plane;
                ssd->superblock[SB].last_update_time = ssd->current_time - ssd->superblock[SB].update_time;
                for(int ch = 0; ch< ssd->parameter->channel_number; ch++){
                    ssd->superblock[SB].invalid_page_count += ssd->channel_head[ch].chip_head[SB_chip].die_head[0].plane_head[0].blk_head[SB_block].invalid_page_num;
                }
                float diffB;
                for(int ch = 0; ch< ssd->parameter->channel_number; ch++){
                    float Avg = ssd->superblock[SB].invalid_page_count/8;
                    diffB += (Avg-ssd->channel_head[ch].chip_head[SB_chip].die_head[die].plane_head[plane].blk_head[SB_block].invalid_page_num)*(Avg-ssd->channel_head[ch].chip_head[SB_chip].die_head[die].plane_head[plane].blk_head[SB_block].invalid_page_num);
                }
                ssd->superblock[SB].Invaild_variance = diffB/8;
                if(SB == 1023){
                    ssd->superblock[SB].neighbor_update = ssd->superblock[SB-1].window_udpate_count;
                }else{
                    ssd->superblock[SB].neighbor_update = ssd->superblock[SB+1].window_udpate_count+last_SB_update;
                }
                    fprintf(ssd->warm_file, "%d,", SB);
                    fprintf(ssd->warm_file, "%ld,",ssd->superblock[SB].last_update_time);
                    fprintf(ssd->warm_file, "%d,",ssd->superblock[SB].window_udpate_count);
                    fprintf(ssd->warm_file, "%d,",ssd->superblock[SB].invalid_page_count);
                    fprintf(ssd->warm_file, "%3f,", ssd->superblock[SB].Invaild_variance);
                    fprintf(ssd->warm_file, "%ld,", ssd->superblock[SB].Avg_update_time);
                    fprintf(ssd->warm_file, "%d\n", ssd->superblock[SB].neighbor_update);
                last_SB_update = ssd->superblock[SB].window_udpate_count;
                ssd->superblock[SB].window_udpate_count = 0;
                ssd->superblock[SB].Avg_update_time = 0;
                ssd->superblock[SB].invalid_page_count = 0;
            }
        }

        ssd->data_window = ssd->parameter->block_plane*ssd->parameter->page_block*(ssd->parameter->gc_threshold-ssd->parameter->gc_hard_threshold);
        ssd->is_update_flag = 0;
    }
    return ssd;
}