

def load_my_state_dict(model, checkpoint):
    own_state = model.state_dict()
    for name, param in checkpoint.items():
        if name not in own_state:
            tmp = name.split('.')
    
            if tmp[0] == 'bi_lstm_stack':
                new_name = f"{'.'.join(tmp[:4])}.{tmp[5]}_l0"
                if tmp[4].split('_')[-1] == 'b':
                    new_name += '_reverse'
            elif tmp[0] == 'pick_lstms':
                new_name = f"{'.'.join(tmp[:2])}.{tmp[-1]}_l0"
            name = new_name
        
        param = param.data
        own_state[name].copy_(param)
        
    model.load_state_dict(own_state)
    return model
