#from mvranalysis import loadMazeMat
#from mvranalysis import load_session
#from mvranalysis import correctL
import mvranalysis as mvr 


path = '/Volumes/freeman/Nick/mVR/sessions/000038'
maze = mvr.loadMazeMat(path + '/behavior')
session = mvr.load_session(path)
s = mvr.correctL(session)




#print session.xMazePos[632797]

#trials = filter_trials(trials, type=1)
#draw_trials(trials, range=[5,90])
