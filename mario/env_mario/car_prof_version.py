import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import gym

# Defina as funções necessárias para encontrar o erro e calcular o controle PID
def green_mask(observation):
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

    ## recorte o verde
    imask_green = mask_green > 0
    green = np.zeros_like(observation, np.uint8)
    green[imask_green] = observation[imask_green]
    return green

def gray_scale(observation):
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    return gray

def blur_image(observation):
    blur = cv2.GaussianBlur(observation, (5, 5), 0)
    return blur

def canny_edge_detector(observation):
    canny = cv2.Canny(observation, 50, 150)
    return canny

def find_error(observation, previous_error, plot=False):
    cropped = observation[63:65, 24:73]
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(observation)
        axs[0, 0].set_title('Original')
        axs[0, 1].imshow(green_mask(cropped))
        axs[0, 1].set_title('Green Mask')
        axs[1, 0].imshow(gray_scale(green_mask(cropped)), cmap='gray')
        axs[1, 0].set_title('Gray Scale')
        axs[1, 1].imshow(canny_edge_detector(blur_image(gray_scale(green_mask(cropped)))), cmap='gray')
        axs[1, 1].set_title('Canny Edge Detector')
        plt.show()
    green = green_mask(cropped)
    gray = gray_scale(green)
    blur = blur_image(gray)
    canny = canny_edge_detector(blur)

    # encontre todos os valores não nulos na faixa recortada
    # Esses pontos não nulos (pixels brancos) correspondem às bordas da estrada
    nz = cv2.findNonZero(canny)

    # coordenadas horizontais do centro da estrada na fatia recortada
    mid = 24

    # alguns ajustes adicionais obtidos por meio de tentativa e erro
    if nz[:,0,0].max() == nz[:,0,0].min():
        if nz[:,0,0].max() < 30 and nz[:,0,0].max() > 20:
            return previous_error
        if nz[:,0,0].max() >= mid:
            return -15
        else:
            return 15
    else:
        return ((nz[:,0,0].max() + nz[:,0,0].min()) / 2) - mid

def pid(error, previous_error):
    Kp = 0.02
    Ki = 0.03
    Kd = 0.2

    steering = Kp * error + Ki * (error + previous_error) + Kd * (error - previous_error)

    return steering

# Crie o ambiente e comece a executar
env = gym.make('CarRacing-v2', render_mode="human")
observation = env.reset()
env.render() 

rewardsum = 0  
previous_error = 0    

for x in [1,0] * 500:     
    try:
        error = find_error(observation, previous_error)
    except:
        error = -15
        print("error")
        pass

    steering = pid(error, previous_error)
    action = (steering, x, 0)

    observation, reward, terminated, truncated, info = env.step(action)
    previous_error = error
    rewardsum += reward

    if terminated or truncated:
        observation, info = env.reset()
        break
    
print("recompensa", rewardsum)
