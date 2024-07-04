# --------------------------------------------------
#
#     Copyright (C) 2024


CEND      = '\33[0m'
CBOLD     = '\33[1m'
CITALIC   = '\33[3m'
CURL      = '\33[4m'
CBLINK    = '\33[5m'
CBLINK2   = '\33[6m'
CSELECTED = '\33[7m'

CBLACK  = '\33[30m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE  = '\33[36m'
CWHITE  = '\33[37m'

CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

CGREY    = '\33[90m'
CRED2    = '\33[91m'
CGREEN2  = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2   = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2  = '\33[96m'
CWHITE2  = '\33[97m'

CGREYBG    = '\33[100m'
CREDBG2    = '\33[101m'
CGREENBG2  = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2   = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2  = '\33[106m'
CWHITEBG2  = '\33[107m'



def dicom_settings(dicom_config):

    settings = {}

    settings['all_image_folder'] = dicom_config.get('dicominputfolders', 'all_image_folder')
    settings['training_folder'] = dicom_config.get('dicominputfolders', 'training_folder')
    settings['cross_validation_folder'] = dicom_config.get('dicominputfolders', 'cross_validation_folder')
    settings['training_label'] = dicom_config.get('dicominputfolders', 'training_label')
    settings['cross_validation_label'] = dicom_config.get('dicominputfolders', 'cross_validation_label')
    settings['test_onthefly'] = dicom_config.get('dicominputfolders', 'test_onthefly')
    settings['inference_folder'] = dicom_config.get('dicominputfolders', 'inference_folder')


    settings['debug'] = dicom_config.get('dicominputparameters', 'debug')
    settings['name'] = dicom_config.get('dicominputparameters', 'name')
    settings['epochs'] = dicom_config.getint('dicominputparameters', 'epochs')
    settings['batch_size'] = dicom_config.getint('dicominputparameters', 'batch_size')
    settings['net_verbose'] = dicom_config.getint('dicominputparameters', 'net_verbose')
    settings['gpu_number'] = dicom_config.getint('dicominputparameters', 'gpu_number')
    settings['threshold'] = dicom_config.getfloat('dicominputparameters',
                                                   'threshold')
    settings['error_tolerance'] = dicom_config.getfloat('dicominputparameters',
                                                        'error_tolerance')

    settings['learning_rate'] = dicom_config.getfloat('dicominputparameters',
                                                   'learning_rate')

    settings['weight_decay'] = dicom_config.getfloat('dicominputparameters',
                                                   'weight_decay')


    keys = list(settings.keys())
    for k in keys:
        value = settings[k]
        if value == 'True':
            settings[k] = True
        if value == 'False':
            settings[k] = False

    return settings

def all_settings_show(settings):
    print('\x1b[6;30;44m' + '                   ' + '\x1b[0m')
    print('\x1b[6;30;44m' + '   All settings!   ' + '\x1b[0m')
    print('\x1b[6;30;44m' + '                   ' + '\x1b[0m')
    print(" ")
    keys = list(settings.keys())
    for key in keys:
        print(CRED + key, ':' + CEND, settings[key])
    print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')