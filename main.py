# import main
from pygame.locals import *
from utils import *
from initializers import *

fpsClock = pg.time.Clock()
pg.init()
window = pg.display.set_mode((WINDOW_WIDTH , WINDOW_HEIGHT))
pg.display.set_caption("Catch The Ball")

# rect = pg.Rect(RECT_LEFT , RECT_TOP , RECT_WIDTH , RECT_HEIGHT)

font = pg.font.Font(None , 30)
episode_rewards = []
agent = DQNAgent()
for episode in range(1,EPISODES):
    agent.tensorboard.step = i
    if not episode%SHOW_EVERY:
        SHOW = True
    else:
        SHOW = False
    state = reset()
    episode_reward = 0
    step = 1
    done = False
    while not done:
        if np.random.random() < EPSILON:
            action = np.random.randint(0 , ENV_OUTPUT)
        else:
            action = np.argmax(agent.get_qs(state))
        # action = agent.get_qs(state)
        new_state , reward , done = new_state_function(state ,action)
        episode_reward += reward
        if reward ==CATCH_REWARD :
            score+=1
            score_every+=1
        elif reward == -LOOSE_PENALTY:
            missed+=1
        # agent.update_replay_memory(((state.rect.left/WINDOW_WIDTH, state.circle.circleX/WINDOW_WIDTH , state.circle.circleY/WINDOW_HEIGHT) , action , reward , (new_state.rect.left, new_state.circle.circleX , new_state.circle.circleY) , done))
        agent.update_replay_memory(((state.rect.left/WINDOW_WIDTH, state.circle.circleX/WINDOW_WIDTH , state.circle.circleY/WINDOW_HEIGHT) ,
         action , reward , (new_state.rect.left/WINDOW_WIDTH, new_state.circle.circleX /WINDOW_WIDTH, new_state.circle.circleY/WINDOW_HEIGHT) , done))
        
        agent.train(done, step)
        if SHOW:
            window.fill(WHITE)
            
            pg.draw.circle(window , RED , (state.circle.circleX , int(state.circle.circleY)) , CIRCLE_RADIUS)
            pg.draw.rect(window , GREEN , (state.rect.left , state.rect.top , state.rect.width , state.rect.height ))
            text = font.render('Score: '+str(score) , True , (238,23,122))
            text1 = font.render('Missed: '+str(missed) , True , (238,23,122))
            text2 = font.render('Success Score: '+str(round(score/episode , 4)*100)+' %' , True ,  (238,23,122))
            text3 = font.render('Success Every: '+str(round(score_every/SHOW_EVERY ,4)*100)+' %' , True ,(238,23,122) )
            window.blit(text , (WINDOW_WIDTH -120 , 10))
            window.blit(text1 , (WINDOW_WIDTH -280 , 10))
            window.blit(text2 , (WINDOW_WIDTH -280 , 32))
            window.blit(text3 , (WINDOW_WIDTH -280 , 50))
            pg.display.update()
            fpsClock.tick(FPS)

        state = new_state
        step+=1
    episode_rewards.append((episode_reward , EPSILON ,round(score/episode,4)))
    if SHOW:
        score_every=0
    # print("For Epiode " , i , " the reward is ",episode_reward , " and epsilon is ",EPSILON)
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
        EPSILON = max(MIN_EPSILON , EPSILON)

np.save('rewards/Episode_rewards'+str(round(score/episode ,3))+'-time-'+str(time.time())+'.npy' , episode_rewards)
        # print(np.argmax(agent.get_qs(s)))
# while True : 
#     agent.tensorboard.step = i
#     for event in pg.event.get():
#         if event.type == QUIT:
#             pg.quit()
    
#     window.fill(WHITE)

#     if CIRCLE_CENTER_Y >= WINDOW_HEIGHT - RECT_HEIGHT - CIRCLE_RADIUS:
#         reward = calculate_score(rect , Circle(CIRCLE_CENTER_X , CIRCLE_CENTER_Y ))
#         CIRCLE_CENTER_X = circle_falling( CIRCLE_RADIUS )
#         CIRCLE_CENTER_Y = 50
#     else:
#         reward = 0
#         CIRCLE_CENTER_Y += CIRCLE_Y_STEP_FALLING

#     s = State(rect , Circle(CIRCLE_CENTER_X , CIRCLE_CENTER_Y))
#     action = get_best_score(s)
#     r0 = calculate_score(s.rect , s.circle)
#     s1 = new_state_after_action(s , action)
#     # print(Q[state_to_number(s),action])
#     Q[state_to_number(s) , action] += lr*(r0+ y*np.max(Q[state_to_number(s1), :])) -  Q[state_to_number(s),action]
#     #  lr*(r0+ y*np.max[Q[state_to_number(s1), :]]) - 
#     rect = new_rect_after_action(s.rect , action)
#     CIRCLE_CENTER_X = s.circle.circleX 
#     CIRCLE_CENTER_Y = int(s.circle.circleY)
#     # print(RED , CIRCLE_CENTER_X , CIRCLE_CENTER_Y,CIRCLE_RADIUS)
#     # print(rect)
#     pg.draw.circle(window , RED , (CIRCLE_CENTER_X , CIRCLE_CENTER_Y) , CIRCLE_RADIUS)
#     pg.draw.rect(window , GREEN , rect)

#     if reward ==1 :
#         score+=reward
#     elif reward == -1:
#         missed+=reward

#     text = font.render('Score: '+str(score) , True , (238,23,122))
#     text1 = font.render('Missed: '+str(missed) , True , (238,23,122))
#     window.blit(text , (WINDOW_WIDTH -120 , 10))
#     window.blit(text1 , (WINDOW_WIDTH -280 , 10))
#     pg.display.update()
#     fpsClock.tick(FPS)

#     if i== 10000:
#         break
#     else:
#         i+=1