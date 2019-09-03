from classes import *
from initializers import *
import random
import pygame as pg

def new_state_after_action(s , action):
    rect = None
    if action ==2:
        if s.rect.right + s.rect.width > WINDOW_WIDTH:
            rect = s.rect
        else:
            rect = pg.Rect(s.rect.left + RECT_WIDTH_MOVE , s.rect.top , 
            s.rect.width , s.rect.height)
    elif action ==1:
        if s.rect.left-s.rect.width<0:
            rect = s.rect
        else:
            rect = pg.Rect(s.rect.left - RECT_WIDTH_MOVE , s.rect.top , 
            s.rect.width , s.rect.height)
    else:
        rect  = s.rect
    newCircle = Circle(s.circle.circleX , s.circle.circleY + CIRCLE_Y_STEP_FALLING)

    return State(rect , newCircle)

def new_rect_after_action(rect , action):
    if action == 2:
        if rect.right + rect.width > WINDOW_WIDTH:
            return rect
        else:
            return pg.Rect(rect.left + RECT_WIDTH_MOVE, rect.top , rect.width , rect.height)

    elif action == 1:
        if rect.left - rect.width <  0:
            return rect
        else:
            return pg.Rect(rect.left - RECT_WIDTH_MOVE , rect.top , rect.width , rect.height)
    else:
        return rect 

def new_state_function(state , action):
    if state.circle.circleY >= WINDOW_HEIGHT - state.rect.height - CIRCLE_RADIUS:
        done = True
        rect  = state.rect
        newCircle = Circle(state.circle.circleX , state.circle.circleY + CIRCLE_Y_STEP_FALLING)
        if state.rect.left <= state.circle.circleX <= state.rect.left + state.rect.width:   
            reward = CATCH_REWARD
        else:
            reward = -LOOSE_PENALTY
    else:
        done = False
        newCircle = Circle(state.circle.circleX , state.circle.circleY + CIRCLE_Y_STEP_FALLING)
        if action == 2:
            reward = -MOVE_PENALTY
            if state.rect.left + state.rect.width + RECT_WIDTH_MOVE > WINDOW_WIDTH:
                rect = state.rect
            else:
                rect = Rect(state.rect.left + RECT_WIDTH_MOVE , state.rect.top , 
                state.rect.width , state.rect.height)
        elif action ==1:
            reward = -MOVE_PENALTY
            if state.rect.left-RECT_WIDTH_MOVE < 0:
                rect = state.rect
            else:
                rect = Rect(state.rect.left - RECT_WIDTH_MOVE , state.rect.top , 
                state.rect.width , state.rect.height)     
        elif action == 0 :
            reward = 0
            rect  = state.rect
    return State(rect , newCircle) , reward , done            
    

def circle_falling(circle_radius):
    newx = 100 - circle_radius
    multiplier = random.randint(1,8)
    newx *= multiplier
    return newx

def reset():
    rect  = Rect(RECT_LEFT , RECT_TOP , RECT_WIDTH , RECT_HEIGHT)
    newCircle = Circle(random.randint(CIRCLE_RADIUS+1 , WINDOW_WIDTH - CIRCLE_RADIUS-1), CIRCLE_CENTER_Y)
    return State(rect , newCircle)

def calculate_score(rect , circle):
    if rect.left <= circle.circleX <= rect.right:
        return 1 
    else:
        return -1

def state_to_number(s):
    r = s.rect.left
    c = s.circle.circleY
    n = int(float(str(r) + str(c) + str(s.circle.circleX)))

    if n in QIDic:
        return QIDic[n]
    else:
        if len(QIDic):
            maximum = max(QIDic , key = QIDic.get)
            QIDic[n] = QIDic[maximum] + 1
        else:
            QIDic[n] = 1
    return QIDic[n]

def get_best_score(s):
    return np.argmax(Q[state_to_number(s), : ])