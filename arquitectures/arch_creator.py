from .Kamnitsas import generate_kamnitsas_model
from .Dolz import generate_dolz_multi_model
from .Cicek import generate_unet_model
from .Guerrero import generate_uresnet_model

def generate_model(gen_conf, train_conf) :
    approach = train_conf['approach']

    if approach == 'Kamnitsas' :
        return generate_kamnitsas_model(gen_conf, train_conf)
    if approach == 'DolzMulti' :
        return generate_dolz_multi_model(gen_conf, train_conf)
    if approach == 'Cicek' :
        return generate_unet_model(gen_conf, train_conf)
    if approach == 'Guerrero' :
        return generate_uresnet_model(gen_conf, train_conf)

    return None
