#include <resize_function.h>
#include <stdint.h>
#include <stdio.h>

int compute_letter_box(LETTER_BOX *lb)
{
    lb->img_wh_ratio = (float)lb->in_width / (float)lb->in_height;
    lb->target_wh_ratio = (float)lb->target_width / (float)lb->target_height;

    if (lb->img_wh_ratio >= lb->target_wh_ratio)
    {
        // pad height dim
        lb->resize_scale_w = (float)lb->target_width / (float)lb->in_width;
        lb->resize_scale_h = lb->resize_scale_w;

        lb->resize_width = lb->target_width;
        lb->w_pad_left = 0;
        lb->w_pad_right = 0;

        lb->resize_height = (int)((float)lb->in_height * lb->resize_scale_h);
        lb->h_pad_top = (lb->target_height - lb->resize_height) / 2;
        if (((lb->target_height - lb->resize_height) % 2) == 0)
        {
            lb->h_pad_bottom = lb->h_pad_top;
        }
        else
        {
            lb->h_pad_bottom = lb->h_pad_top + 1;
        }
    }
    else
    {
        // pad width dim
        lb->resize_scale_h = (float)lb->target_height / (float)lb->in_height;
        lb->resize_scale_w = lb->resize_scale_h;

        lb->resize_width = (int)((float)lb->in_width * lb->resize_scale_w);
        lb->w_pad_left = (lb->target_width - lb->resize_width) / 2;
        if (((lb->target_width - lb->resize_width) % 2) == 0)
        {
            lb->w_pad_right = lb->w_pad_left;
        }
        else
        {
            lb->w_pad_right = lb->w_pad_left + 1;
        }

        lb->resize_height = lb->target_height;
        lb->h_pad_top = 0;
        lb->h_pad_bottom = 0;
    }
    return 0;
}

inline static int clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}

int h_reverse(int h, LETTER_BOX lb)
{
    if (!lb.reverse_available)
    {
        return clamp(h, 0, lb.in_height);
    }
    int r_h = clamp(h, 0, lb.target_height) - lb.h_pad_top;
    r_h = clamp(r_h, 0, lb.resize_height);
    r_h = r_h / lb.resize_scale_h;
    r_h = clamp(r_h, 0, lb.in_height);
    return r_h;
}

int w_reverse(int w, LETTER_BOX lb)
{
    if (!lb.reverse_available)
    {
        return clamp(w, 0, lb.in_width);
    }
    int r_w = clamp(w, 0, lb.target_width) - lb.w_pad_left;
    r_w = clamp(r_w, 0, lb.resize_width);
    r_w = r_w / lb.resize_scale_w;
    r_w = clamp(r_w, 0, lb.in_width);
    return r_w;
}