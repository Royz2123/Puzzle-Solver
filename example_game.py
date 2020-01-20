import pygame

# --- constants --- (UPPER_CASE names)

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED   = (255,   0,   0)

FPS = 30

# --- classses --- (CamelCase names)

# empty

# --- functions --- (lower_case names)

# empty

# --- main ---

# - init -

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#screen_rect = screen.get_rect()

pygame.display.set_caption("Puzzle Solver")

# - objects -

rectangle = pygame.rect.Rect(176, 134, 17, 17)
rectangle_draging = False


class ImagePiece(pygame.sprite.Sprite):

    def __init__(self, pos=(0, 0)):
        super(ImagePiece, self).__init__()
        path = r'.\\image_processing\\pieces\\game_images\\Piece 0.png'
        self.original_image = pygame.image.load(path).convert()
        self.image = self.original_image  # This will reference our rotated image.
        self.rect = self.image.get_rect().move(pos)
        self.angle = 0
        self.chosen=False

    def update(self, angle):
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.angle += angle % 360  # Value will reapeat after 359. This prevents angle to overflow.

image = ImagePiece()
all_sprites_list = pygame.sprite.Group()
all_sprites_list.add(image)

# - mainloop -

clock = pygame.time.Clock()

running = True
rotating = 0

while running:

    # - events -

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if rectangle.collidepoint(event.pos):
                    rectangle_draging = True
                    mouse_x, mouse_y = event.pos
                    offset_x = rectangle.x - mouse_x
                    offset_y = rectangle.y - mouse_y

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                rectangle_draging = False

        elif event.type == pygame.MOUSEMOTION:
            if rectangle_draging:
                mouse_x, mouse_y = event.pos
                rectangle.x = mouse_x + offset_x
                rectangle.y = mouse_y + offset_y

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                rotating = 1
            if event.key == pygame.K_RIGHT:
                rotating = -1

        elif event.type == pygame.KEYUP:  # Added keyup
            if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                rotating = 0

    if rotating != 0:
        image.update(rotating)



                # - updates (without draws) -

    # empty

    # - draws (without updates) -

    screen.fill(WHITE)

    pygame.draw.rect(screen, RED, rectangle)
    all_sprites_list.draw(screen)

    pygame.display.flip()

    # - constant game speed / FPS -

    clock.tick(FPS)

# - end -

pygame.quit()