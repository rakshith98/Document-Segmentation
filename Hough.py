import cv2
import numpy as np
import math
import random
import operator

from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections

def angle_between_points( p0, p1, p2 ):
	#print(p0,p1,p2)
	a = (p1[0][0]-p0[0][0])**2 + (p1[0][1]-p0[0][1])**2
	b = (p1[0][0]-p2[0][0])**2 + (p1[0][1]-p2[0][1])**2
	c = (p2[0][0]-p0[0][0])**2 + (p2[0][1]-p0[0][1])**2
	#print(a,b,c)
	if a==0 or b==0:
		return 1
	else:
		return math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/(22/7)

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area







img = cv2.imread('q.png')#uoppp.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
#edges = cv2.GaussianBlur(edges,(5,5),0)
lines = cv2.HoughLines(edges,1,np.pi/180,150)
print(lines)
val  = sorted(lines, key = lambda x: (x[0][1], x[0][0]))
#val  = sorted(lines, key = lambda x: (x[0][0], x[0][1]))
'''
for line in val:
    for rho,theta in line:
        #print(line)
        #val.append(rho/theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
'''
#val.sort()
#print(val)


threshold = 10
new=[]
base1 = val[0][0][0]
base2 = val[0][0][1]
b=val[0]


#print(val)

qqp=[]
for i in val:
    count=0
    for j in val:
        if(abs(j[0][0]-i[0][0])<20 and abs(j[0][1]-i[0][1])<.3):
            count+=1
    qqp.append([count,i])        

qqpw  = sorted(qqp, key = lambda x: (x[0]))

#print(qqpw)
for i in qqp:
    if(i[0] > 0):
        if(abs(base1 -i[1][0][0])>20):
           #print(j)
           new.append(b)
           b=i[1]
           base1 = i[1][0][0]
           base2 = i[1][0][1]
new.append(i[1])
      
segmented = segment_by_angle_kmeans(new)


intersections = segmented_intersections(segmented)
'''
#print(intersections)
print(segmented)
print(new)
for line in new:
    for rho,theta in line:
        #print(line)
        #val.append(rho/theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1500*(-b))
        y1 = int(y0 + 1500*(a))
        x2 = int(x0 - 1500*(-b))
        y2 = int(y0 - 1500*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
''' 
print(len(intersections))       
for point in intersections:
    #print(point)
    cv2.circle(img,tuple(point[0]),7,(0,255,0),-1)
    print(type(point[0]))

intersection=[]
three_points = []
for i in range(0,len(intersections)):
	one = intersections[i]
	for j in range(i+1,len(intersections)):
		two = intersections[j]
		for k in range(j+1,len(intersections)):
			three = intersections[k]
			a = angle_between_points(one,two,three)
			if a>85 and a<95:
				three_points.append([one,two,three])

four_points = []
for i in three_points:
	for j in range(0,len(intersections)):
		if intersections[j]!= i[0] or intersections[j]!=i[1] or intersections[j]!=i[2]:
			four_points.append([i[0],i[1],i[2],intersections[j]])
			
print(len(four_points))
print(len(three_points))
print("_")
			



	
final_box = []		
for points in four_points:
	
	lt= angle_between_points(points[0],points[1],points[2])
	lb = angle_between_points(points[3],points[0],points[1])
	rt = angle_between_points(points[1],points[2],points[3])
	rb = angle_between_points(points[2],points[3],points[0])
	if (lt>85 and lt<95) and (lb>85 and lb<95) and (rt>85 and rt<95) and (rb>85 and rb<95):
		final_box.append(points)

area = []
for point in final_box:
	a = [(point[0][0][0],point[0][0][1]),(point[1][0][0],point[1][0][1]),(point[2][0][0],point[2][0][1]),(point[3][0][0],point[3][0][1])]
	area.append(PolygonArea(a))

print(len(area))	
index, value = max(enumerate(area), key=operator.itemgetter(1))

def display(point):
	print(point)
	tr_img = img
	color = (random.choice(range(256)),random.choice(range(256)),random.choice(range(256)))
	#print(color)
	#print(type(color[0]))
	cv2.line(img,(point[0][0][0],point[0][0][1]),(point[1][0][0],point[1][0][1]),color,2)
	cv2.line(img,(point[1][0][0],point[1][0][1]),(point[2][0][0],point[2][0][1]),color,2)
	cv2.line(img,(point[2][0][0],point[2][0][1]),(point[3][0][0],point[3][0][1]),color,2)
	cv2.line(img,(point[3][0][0],point[3][0][1]),(point[0][0][0],point[0][0][1]),color,2)
	cv2.imshow('houghlines3.jpg',tr_img)
	cv2.waitKey(0)
		
display(final_box[index])
        
#print(new)
'''
print(len(lines),len(new))
cv2.imshow('houghlines3.jpg',img)
cv2.imwrite("houghlines.jpg",img)
cv2.waitKey(0)
'''
