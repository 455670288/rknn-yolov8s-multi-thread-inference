#pragma once

struct LETTER_BOX
{
    int in_width = 0, in_height = 0;
    int target_width = 0, target_height = 0;
    int channel = 3;

    float img_wh_ratio = 1, target_wh_ratio = 1;
    float resize_scale_w = 0, resize_scale_h = 0;
    int resize_width = 0, resize_height = 0;
    int h_pad_top = 0, h_pad_bottom = 0;
    int w_pad_left = 0, w_pad_right = 0;

    bool reverse_available = false;
};

int compute_letter_box(LETTER_BOX *lb);

int h_reverse(int h, LETTER_BOX lb);

int w_reverse(int w, LETTER_BOX lb);