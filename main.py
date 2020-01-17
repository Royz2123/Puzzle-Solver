import image_processing.image_proc as im_proc
import image_processing.util as util
import image_processing.camera as camera
import time
import algorithmics.puzzle as puzzle
import cv2
import mechanics.mechanics_api as mechanics_api


TEST_MODE = True

# Exciting! actual real main
# Need to think about shceme - maybe we need thread for greedy in terms of speed
# If it's slow, then greedy and execution are intertwined

def showImage(name,image):
    img_show = cv2.resize(image, None, fx=0.2, fy=0.2)
    cv2.imshow(name,img_show)

def get_images(test=True):
    if test:
        above, below = util.get_test_images()
    else:
        mechanics_api.send_command_accel(l_on=1)
        time.sleep(1)

        for i in range(5):
            below = camera.take_picture(still_camera=True)
        below = cv2.cvtColor(below, cv2.COLOR_BGR2GRAY)

        mechanics_api.send_command_accel(l_on=0)
        time.sleep(0.5)
        for i in range(5):
            above = camera.take_picture(still_camera=True)

    above = util.relevant_section(above, test)
    below = util.relevant_section(below, test)
    return above, below


def blinking_lights():
    if not TEST_MODE:
        for i in range(10):
            mechanics_api.send_command_accel(l_on=1)
            time.sleep(0.1)
            mechanics_api.send_command_accel(l_on=0)
            time.sleep(0.1)


def main():
    blinking_lights()

    # Parse pieces
    above, below = get_images(test=TEST_MODE)

    showImage("above", above)
    # cv2.waitKey(0)
    # cv2.imwrite("mechanics/callibration/test.jpg", above)
    showImage("below", below)

    pieces = im_proc.get_pieces(above, below, TEST_MODE)
    puzzle_obj = puzzle.Puzzle(pieces)

    # Build puzzle
    puzzle_obj.greedy()
    puzzle_obj.connect()
    puzzle_obj.display()

    # Execute puzzle
    # command_list = puzzle_obj.create_command_list()
    # command_list = [(1400, 378, 1960, 200, 5.048790501062726),
    #                 (393, 917, 1960, 576, 0.6189205831106102),
    #                 (1417, 895, 2160, 200, 0.9539093029212884),
    #                 (851, 958, 2160, 576, 5.22494859553469),
    #                 (921, 363, 2360, 200, -0.6820334291744992),
    #                 (441, 329, 2360, 576, 2.203894689919861)]
    # command_list = [(1793-1496,453,1793-1000,453, 0)]
    #

    # Execute puzzle
    command_list = puzzle_obj.create_command_list()
    print(command_list)
    mechanics_api.execute_command_accel(command_list)
    blinking_lights()

if __name__ == "__main__":
    main()

