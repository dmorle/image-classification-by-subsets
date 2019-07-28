import pygame
from hilbertCurve import *
from LinearAlg import *
from pygame.locals import *

def main():
	width = 600
	height = 600

	size = (width, height)

	hC = hilbertCurve(3)
	curve = hC.getDPoints()
	for i in range(len(curve)):
		curve[i] = VctSum(PWProd(curve[i], (350, -350)), (150, 450))

	pygame.init()

	screen = pygame.display.set_mode(size)
	pygame.display.set_caption("Hibert Curve Testing")

	running = True
	clock = pygame.time.Clock()

	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 1:
					mouseLoc = pygame.mouse.get_pos()
					#left click
				if event.button == 3:
					mouseLoc = pygame.mouse.get_pos()
					#right click
		pygame.draw.lines(screen, (255, 255, 255), False, curve)
		pygame.display.flip()
		clock.tick(60)
	pygame.quit()
	return

if __name__ == '__main__':
	main()