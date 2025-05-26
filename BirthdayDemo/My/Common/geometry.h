#pragma once 

#include <stdio.h> 
#include <math.h> 

#define PI  3.142 

// function declarations 
void drawSphere(float cx, cy, float r); 
void drawSemiCircle(float cx, float cy, float radius, float start_angle, float end_angle, 
                float red, float green, float blue); 
void drawFilledCirveWithGivenAngle(float cx, float cy, float r, float start_angle, float end_angle); 
void circle(float r); 

