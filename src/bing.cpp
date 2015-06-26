#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include <boost/python.hpp>
#include <boost/numpy.hpp>

namespace py = boost::python;
namespace np = boost::numpy;
namespace sa = cv::saliency;

class BING {
private:

  cv::saliency::ObjectnessBING *binger;
  std::vector<cv::Vec4i> saliency_map;

public:

  BING(const std::string& trained_model,
       const int        & base_window_size_quantization,
       const int        & window_size,
       const int        & non_maximal_supress_size) {
    binger = new cv::saliency::ObjectnessBING();
    binger->setTrainingPath(trained_model);
    binger->setBase(base_window_size_quantization);
    binger->setW(window_size);
    binger->setNSS(non_maximal_supress_size);
  }

  ~BING() {
    delete binger;
  }

  np::ndarray objectness(const np::ndarray& img) {
    cv::Mat image = convert_to_cvmat(img);

    // minX, minY, maxX, maxY
    binger->computeSaliency(image, saliency_map);
    std::vector<float> objectness_value = binger->getobjectnessValues();
    py::tuple   shape = py::make_tuple(saliency_map.size(), 5);
    np::dtype   dtype  = np::dtype::get_builtin<float>();
    np::ndarray result = np::zeros(shape, dtype);
    float *data        = reinterpret_cast<float *>(result.get_data());
    assert(objectness_value.size() == saliency_map.size());
    for (size_t i = 0; i < saliency_map.size(); ++i) {
      data[i * 5 + 0] = saliency_map[i][0];
      data[i * 5 + 1] = saliency_map[i][1];
      data[i * 5 + 2] = saliency_map[i][2];
      data[i * 5 + 3] = saliency_map[i][3];
      data[i * 5 + 4] = objectness_value[i];
    }

    return result;
  }

  np::ndarray proposals(
    const np::ndarray& img, const int& num,
    const int& height, const int& width, const int& channels, const int& pad) {
    const int   dim    = height * width * channels;
    cv::Mat     image  = convert_to_cvmat(img);
    py::tuple   shape  = py::make_tuple(num, height, width, channels);
    np::dtype   dtype  = np::dtype::get_builtin<float>();
    np::ndarray result = np::zeros(shape, dtype);
    float *data        = reinterpret_cast<float *>(result.get_data());
    const int max_num  = num < saliency_map.size() ? num : saliency_map.size();

    for (size_t i = 0; i < max_num; ++i) {
      const int min_x = saliency_map[i][0] - pad >= 0 ?
                        saliency_map[i][0] - pad : 0;
      const int min_y = saliency_map[i][1] - pad >= 0 ?
                        saliency_map[i][1] - pad : 0;
      const int max_x = saliency_map[i][2] + pad < image.cols ?
                        saliency_map[i][2] + pad : image.cols;
      const int max_y = saliency_map[i][3] + pad < image.rows ?
                        saliency_map[i][3] + pad : image.rows;
      cv::Mat patch =
        image(cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y));
      cv::resize(patch, patch, cv::Size(width, height));
      patch.convertTo(patch, CV_32FC(channels));
      float *p_data = reinterpret_cast<float *>(patch.data);
      memcpy(data + dim * i, p_data, dim * sizeof(float));
    }

    return result;
  }

  cv::Mat convert_to_cvmat(const np::ndarray& img) {
    const long *shape   = img.get_shape();
    const int   height  = shape[0];
    const int   width   = shape[1];
    const int   channel = shape[2];
    unsigned char *data = reinterpret_cast<unsigned char *>(img.get_data());
    cv::Mat img_mat(height, width, CV_8UC(channel));

    img_mat.data = data;

    return img_mat;
  }

  template<class T>
  np::ndarray convert_to_ndarray(const cv::Mat& img_mat) {
    py::tuple shape = py::make_tuple(
      img_mat.rows, img_mat.cols, img_mat.channels());
    np::dtype   dtype = np::dtype::get_builtin<T>();
    np::ndarray img   = np::zeros(shape, dtype);
    T *data           = reinterpret_cast<T *>(img.get_data());

    memcpy(data, img_mat.data,
           img_mat.rows * img_mat.cols * img_mat.channels() * sizeof(T));

    return img;
  }
};

BOOST_PYTHON_MODULE(bing) {
  np::initialize();

  py::class_<BING>("BING", py::init<const std::string,
                                    const int,
                                    const int,
                                    const int>())
  .def("objectness", &BING::objectness)
  .def("proposals", &BING::proposals);
}
