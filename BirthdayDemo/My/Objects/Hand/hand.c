#include "hand.h" 

void hand(void) 
{
    glPushMatrix(); 
    {
        glColor3f(1.0f, 1.0f, 1.0f); 
        glScalef(0.6f, 1.0f, 1.0f); 

        glBegin(GL_QUADS); 
        glVertex2f(-0.4f, 0.0f);
        glVertex2f(0.0f, 0.1f);
        glVertex2f(0.0f, 0.0f);
        glVertex2f(-0.4f, -0.1f);
        glEnd(); 
        
        glBegin(GL_QUADS); 
        glVertex2f(0.4f, 0.0f);
        glVertex2f(0.0f, 0.1f);
        glVertex2f(0.0f, 0.0f);
        glVertex2f(0.4f, -0.1f);
        glEnd();

        glBegin(GL_QUADS); 
        glVertex2f(-0.4f, -0.3f);
        glVertex2f(0.0f, -0.32f); 
        glVertex2f(0.0f, -0.42f); 
        glVertex2f(-0.4f, -0.4f); 
        glEnd(); 

        glBegin(GL_QUADS);
        glVertex2f(-0.390f, 0.000f);
        glVertex2f(-0.670f, -0.140f);
        glVertex2f(-0.720f, -0.400f);
        glVertex2f(-0.390f, -0.100f);
        glEnd();

        glBegin(GL_QUADS);
        glVertex2f(-0.4f, -0.300f);
        glVertex2f(-0.400f, -0.400f);
        glVertex2f(-0.720f, -0.400f);
        glVertex2f(-0.55f, -0.220f);
        glEnd(); 

        glBegin(GL_QUADS);
        glVertex2f(-0.670f, -0.140f);
        glVertex2f(-1.670f, -0.320f);
        glVertex2f(-1.670f, -0.690f);
        glVertex2f(-0.720f, -0.400f);
        glEnd(); 

        glVertex2f(0.000f, -0.370f);
        glVertex2f(0.400f, 0.000f);

        glColor3f(1.0f, 1.0f, 1.0f); 
        drawFilledCurveWithGivenAngle(0.4f, 0.0f, 0.1f, 270.0f, 345.0f); 
        drawFilledCurveWithGivenAngle(0.0f, -0.370f, 0.05f, 270.0f, 450.0f); 
    } 
    glPopMatrix(); 
}

