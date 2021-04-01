#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;
//teste
typedef Matx<float, 2, 2> tensor;

//bilateral variables
float sst = 2;
int  gaussian_size = ceil(sst * 2) * 2 + 1;

float sigma_d = 3.0f;
float sigma_r = 4.25f;

float sig_e = 1.0;
float sig_r = 1.6;
float sig_m = 3.0;
float tau = 0.99;

const double PI =  3.1415926f;






static void GetGaussianWeights(float* weights,
	int neighbor,
	float sigma) {
	if ((NULL == weights) || (neighbor < 0))
		return;
	float term1 = 1.0 / (sqrt(2.0 * PI) * sigma);
	float term2 = -1.0 / (2 * pow(sigma, 2));
	weights[neighbor] = term1;
	float sum = weights[neighbor];
	for (int i = 1; i <= neighbor; ++i) {
		weights[neighbor + i] = exp(pow(i, 2) * term2) * term1;
		weights[neighbor - i] = weights[neighbor + i];
		sum += weights[neighbor + i] + weights[neighbor - i];
	}
	// Normalization
	for (int j = 0; j < neighbor * 2 + 1; ++j) {
		weights[j] /= sum;
	}
}

// Prepare 1-d difference of gaussian template.
static void GetDiffGaussianWeights(float* weights,
	int neighbor,
	float sigma_e,
	float sigma_r,
	float tau) {
	if ((NULL == weights) || (neighbor < 0))
		return;
	float* gaussian_e = new float[neighbor * 2 + 1];
	float* gaussian_r = new float[neighbor * 2 + 1];
	GetGaussianWeights(gaussian_e, neighbor, sigma_e);
	GetGaussianWeights(gaussian_r, neighbor, sigma_r);
	float sum = 0;
	for (int i = 0; i < neighbor * 2 + 1; ++i) {
		weights[i] = gaussian_e[i] - tau * gaussian_r[i];
		sum += weights[i];
	}
	// Normalization
	for (int j = 0; j < neighbor * 2 + 1; ++j) {
		weights[j] /= sum;
	}
	delete[] gaussian_e;
	delete[] gaussian_r;
}

Mat color_quantize(Mat src, int s) {
	float step = 1.0 / s;
	vector<float> shades(s);


	for (int i = 0; i < s; i++) {
		shades[i] = step*(i + 0.5);
	}

	Mat m(src.size(), CV_32F);
	for (int r = 0; r < src.rows; r++) {
		float* _s = src.ptr<float>(r);
		float* _m = m.ptr<float>(r);
		for (int c = 0; c < src.cols; c++) {
			int n = floor(_s[c] / step);
			
			if (n == s) //evito sair do limite de tons 
				n -= 1;

			_m[c] = shades[n];
		}
	}
	return m;
}


/*
Mat Custom_Bilateral(Mat src, Mat t, int iterations) {
	const double sigma_e = 1.0;
	const double re = 10 / 255.0;
	const double sigma_g = 0.5;
	const double rg = 10 / 255.0;

	//for compute
	const int alpha = 3 * sigma_e;
	const int beta = alpha;
	const double g_sigma_e = 1.0 / (sqrt(2.0*CV_PI)*sigma_e);
	const double g_re = 1.0 / (sqrt(2.0*CV_PI)*re);
	const double g_sigma_g = 1.0 / (sqrt(2.0*CV_PI)*sigma_g);
	const double g_rg = 1.0 / (sqrt(2.0*CV_PI)*rg);

	//precompute
	vector<float> gs(alpha * 2 + 1);
	vector<float> gt(alpha * 2 + 1);
	for (int i = 0; i < alpha * 2 + 1; i++) {
		float s = i <= alpha ? i : alpha - i;
		gs[i] = g_sigma_e*exp(-(s*s) / (2.0 * sigma_e*sigma_e));
		gt[i] = g_sigma_g*exp(-(s*s) / (2.0 * sigma_g*sigma_g));
	}

	Mat h(src.size(), CV_32F);
	for (int it = 0; it < iterations; it++) {
		for (int r = 0; r < t.rows; r++) {
			for (int c = 0; c < t.cols; c++) {
				Vec2f t0 = t.at<Vec2f>(r, c);
				//If tangent = (0, 0)
				if (t0 == Vec2f(0, 0)) {
					//h.at<float>(r, c) = src.at<float>(r, c)*(g_sigma_c - lo*g_sigma_s)*g_sigma_m;
					h.at<float>(r, c) = src.at<float>(r, c);
				}
				else {
					vector<vector<float>> curve(2 * alpha + 1, vector<float>(2 * beta + 1));
					vector<float> cg_arr(2 * alpha + 1);
					Point cur(c, r);
					float ve = 0;
					float ce = 0;
					//i, travel through the curve center at (c, r)
					for (int i = 0; i <= 2 * alpha; i++) {
						//check if current postion out of image
						if (cur.x < 0 || cur.x >= src.cols || cur.y < 0 || cur.y >= src.rows) {
							//if in positive step, transfer to negative step
							if (i <= alpha) {
								i = alpha;
								Vec2f ti = t.at<Vec2f>(Point(c, r));
								cur = Point(round(c - ti[0]), round(r - ti[1]));
								continue;
							}
							//if in negative step, end of the travel
							else
								break;
						}
						float vg = 0;
						float cg = 0;
						Vec2f ti = t.at<Vec2f>(cur);
						Vec2f gi(ti[1], -ti[0]);
						Point tmp = cur;
						//travel through the perpendicular line 
						for (int j = 0; j <= 2 * beta; j++) {
							//check if current point out of image
							if (tmp.x < 0 || tmp.x >= src.cols || tmp.y < 0 || tmp.y >= src.rows) {
								if (j <= beta) {
									j = beta;
									tmp = Point(round(cur.x - gi[0]), round(cur.y - gi[1]));
									continue;
								}
								else
									break;
							}
							curve[i][j] = gt[j] * (g_rg*exp(-(abs(src.at<float>(tmp) - src.at<float>(cur)) / 2 * rg*rg)));
							vg += curve[i][j];
							cg += src.at<float>(tmp) * curve[i][j];
							if (j < beta)
								tmp = Point(round(tmp.x + gi[0]), round(tmp.y + gi[1]));
							else {
								if (j == beta)
									tmp = cur;
								tmp = Point(round(tmp.x - gi[0]), round(tmp.y - gi[1]));
							}
						}

						cg_arr[i] = cg / vg;
						float tmp_ve = g_sigma_e * (g_re*exp(-(abs(cg_arr[i] - cg_arr[0]) / 2 * re*re)));
						ve += tmp_ve;
						ce += cg_arr[i] * tmp_ve;


						if (i < alpha)
							cur = Point(round(cur.x + ti[0]), round(cur.y + ti[1]));
						else {
							if (i == alpha) {
								cur = Point(c, r);
								ti = t.at<Vec2f>(cur);
							}
							cur = Point(round(cur.x - ti[0]), round(cur.y - ti[1]));
						}
					}
					h.at<float>(r, c) = ce / ve;
				}
			}
		}
		src = h.clone();
	}

	return h;
}
*/


int main(int argc, char *argv[])
{
	std::string input; //= "photo/tomiko2.jpg";
	std::cout << "Insira caminho do arquivo " << std::endl;
	std::getline(std::cin, input);

	Mat im, gray, lab;
	im = imread(input, CV_LOAD_IMAGE_COLOR);

	if (!im.data)
	{
		std::cout << "imagem nao pode ser carregada" << std::endl;
		return -1;
	}

	int rows = im.rows;
	int cols = im.cols;

	cvtColor(im, gray, CV_BGR2GRAY);
	Mat smooth;
	bilateralFilter(gray, smooth, 6, 150, 150);

	Mat dx, dy;
	Sobel(smooth, dx, CV_32F, 1, 0);
	Sobel(smooth, dy, CV_32F, 0, 1);

	Mat jacob; //= Mat::zeros(rows, cols, CV_32FC3);;
	jacob.create(rows, cols, CV_32FC3);

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			float gx = dx.at<float>(i, j);
			float gy = dy.at<float>(i, j);

			jacob.at<Vec3f>(i, j)[0] = gx * gx;
			jacob.at<Vec3f>(i, j)[1] = gy * gy;
			jacob.at<Vec3f>(i, j)[2] = gx * gy;
		}
	}

	GaussianBlur(jacob, jacob, Size2i(gaussian_size, gaussian_size), sst);

	//ETF
	Mat ETF = Mat::zeros(rows, cols, CV_32FC3);
	float E, G, F, lambda, v2x, v2y, v2;

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			E = jacob.at<Vec3f>(i, j)[0];
			G = jacob.at<Vec3f>(i, j)[1];
			F = jacob.at<Vec3f>(i, j)[2];

			lambda = 0.5 * (E + G + sqrtf((G - E)*(G - E) + 4 * F * F));
			v2x = E - lambda;
			v2y = F;
			v2 = sqrtf(v2x * v2x + v2y * v2y);

			ETF.at<Vec3f>(i, j)[0] = (0 == v2) ? 0 : (v2x / v2);

			//ETF.at<Vec3f>(i, j)[0] = (0 == v2) ? 0 : (v2x);

			ETF.at<Vec3f>(i, j)[1] = (0 == v2) ? 0 : (v2y / v2);

			//ETF.at<Vec3f>(i, j)[1] = (0 == v2) ? 0 : (v2y);

			ETF.at<Vec3f>(i, j)[2] = sqrtf(E + G - lambda);
		}
	}


	//BILATERAL FILTER + EDGE EXTRACTION
	Mat FDOG;
	FDOG.create(rows, cols, CV_32FC1);
	//FDOG = 0;

	Mat f0 = Mat::ones(rows, cols, CV_32FC1);
	Mat f1 = Mat::ones(rows, cols, CV_32FC1);

	Mat u1 = Mat::zeros(im.size(), CV_8UC1);

	int near = (int)(ceilf(2 * sig_r));
	float sin, cos;

	float *gauss_w = new float[near * 2 + 1];

	float *sample1, *sample2;

	float sum_diff, sum_dev, sum_1;

	GetDiffGaussianWeights(gauss_w, near, sig_e, sig_r, tau);

	int near2 = ceilf(2 * sig_m);

	float *gauss_w2 = new float[near2 * 2 + 1];

	GetGaussianWeights(gauss_w2, near2, sig_m);
	Mat gr_scale;
	smooth.copyTo(gr_scale);


	// gradient
	for (int i = near; i < (rows - near); ++i)
	{
		for (int j = near; j < (cols - near); ++j)
		{
			cos = ETF.at<Vec3f>(i, j)[1];
			sin = -1 * ETF.at<Vec3f>(i, j)[0];
			sample1 = new float[near * 2 + 1];
			sample1[near] = static_cast<float>(gr_scale.at<uchar>(i, j));
			for (int k = 1; k <= near; ++k)
			{
				int r = round(sin * k);
				int c = round(cos * k);

				sample1[near + k] = static_cast<float>(gr_scale.at<uchar>(i + r, j + c));
				sample1[near - k] = static_cast<float>(gr_scale.at<uchar>(i - r, j - c));
			}

			sum_diff = 0;
			sum_dev = 0;

			for (int k = 0; k < 2 * near + 1; ++k)
			{
				sum_diff += sample1[k] * gauss_w[k];
			}
			f0.at<float>(i, j) = sum_diff;
			delete[] sample1;
		}
	}


	// tangent
	for (int i = near2; i < (rows - near2); ++i)
	{
		for (int j = near2; j < (cols - near2); ++j)
		{
			cos = ETF.at<Vec3f>(i, j)[0];
			sin = ETF.at<Vec3f>(i, j)[1];
			sample2 = new float[near2 * 2 + 1];
			sample2[near2] = f0.at<float>(i, j);

			for (int k = 1; k <= near2; ++k)
			{
				int r = round(sin * k);
				int c = round(cos * k);

				sample2[near2 + k] = f0.at<float>(i + r, j + c);
				sample2[near2 - k] = f0.at<float>(i - r, j - c);
			}

			sum_1 = 0;

			for (int k = 0; k < 2 * near2 + 1; ++k)
			{
				sum_1 += sample2[k] * gauss_w2[k];
			}

			f1.at<float>(i, j) = sum_1;
			if (f1.at<float>(i, j) > 0)
			{
				//u1.at<uchar>(i, j) = 0;
				FDOG.at<float>(i, j) = 255;
			}
			else
			{
				//u1.at<uchar>(i, j) = 255;
				FDOG.at<float>(i, j) = 0;
			}
			delete[] sample2;
		}
	}

	delete[] gauss_w;
	delete[] gauss_w2;




	//QUANTIZATION
	
	Mat lum;
	vector<Mat> lab_channels;
	
	cvtColor(im, lab, COLOR_BGR2Lab);
		
	Mat lab1;
	bilateralFilter(lab, lab1, 6, 150, 150);
	Mat lab2;
	bilateralFilter(lab1, lab2, 6, 150, 150);
	Mat lab3;
	bilateralFilter(lab2, lab3, 6, 150, 150);
	
	split(lab3, lab_channels);
	lab_channels[0].convertTo(lum, CV_32F, 1.0f / 255.0f);

	Mat qt = color_quantize(lum, 8);
	
	qt.convertTo(qt, CV_8U, 255);

	lab_channels[0] = qt;

	Mat qtzd;
	merge(lab_channels, qtzd);

	cvtColor(qtzd, qtzd, COLOR_Lab2BGR);

	
	
	FDOG.convertTo(FDOG, CV_8U, 255);
	cvtColor(FDOG, FDOG, CV_GRAY2BGR);
	
	Mat res;

	//addWeighted(qtzd, 1.0, FDOG, 1.0 , 1.0 , res);
	
	bitwise_and(qtzd, FDOG, res);

	//SHOW



	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", res);
	waitKey(0);


	return 0;
}