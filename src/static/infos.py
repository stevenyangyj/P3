def get_scores(env):

    REF_MIN_SCORE = {
        'halfcheetah' : -280.178953 ,
        'walker2d' : 1.629008 ,
        'hopper' : -20.272305 ,
    }

    REF_MAX_SCORE = {
        'halfcheetah' : 12135.0 ,
        'walker2d' : 4592.3 ,
        'hopper' : 3234.3 ,
    }

    return REF_MIN_SCORE[env], REF_MAX_SCORE[env]