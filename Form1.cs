using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace onnx_test
{
    public partial class Form1 : Form
    {
        static readonly string[] classLabels = new string[]
        {
            "Anthracnose", "Bacterial Canker", "Cutting Weevil",
            "Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
        };
        private string modelPath = "my_model.onnx";  // ✅ Đường dẫn model
        private string imagePath1 = "";       // ✅ Đường dẫn ảnh được chọn
        private string imagePath2 = "";
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            
        }

        private void btnChooseImage1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image files (*.jpg, *.jpeg, *.png)|*.jpg;*.jpeg;*.png";

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                imagePath1 = ofd.FileName;
                pictureBox1.Image = Image.FromFile(imagePath1);
                lblResult1.Text = "Ảnh 1" + Path.GetFileNameWithoutExtension(imagePath1);
            }
        }

        private void btnPredict_Click(object sender, EventArgs e)
        {
            //if (string.IsNullOrEmpty(imagePath1) || !File.Exists(imagePath1))
            if (!File.Exists(modelPath) || !File.Exists(imagePath1) || !File.Exists(imagePath2))
            {
                MessageBox.Show("Hãy chắc chắn bạn đã chọn đủ 2 ảnh và có model.");
                return;
            }
            using (var session = new InferenceSession(modelPath))
            {
                string label1 = PredictImage(session, imagePath1);
                string label2 = PredictImage(session, imagePath2);

                lblResult1.Text = $"Predict Pic1: {label1}";
                lblResult2.Text = $"Predict Pic2: {label2}";
            }
        }
        private string PredictImage(InferenceSession session, string imagePath)
        {
            // Resize ảnh
            Bitmap bitmap = new Bitmap(Image.FromFile(imagePath));
            Bitmap resized = new Bitmap(bitmap, new Size(224, 224));
            var inputTensor = new DenseTensor<float>(new[] { 1, 224, 224, 3 });

            for (int y = 0; y < 224; y++)
            {
                for (int x = 0; x < 224; x++)
                {
                    Color pixel = resized.GetPixel(x, y);
                    inputTensor[0, y, x, 0] = pixel.R;
                    inputTensor[0, y, x, 1] = pixel.G;
                    inputTensor[0, y, x, 2] = pixel.B;
                }
            }
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_layer_1", inputTensor)
            };

            using (var results = session.Run(inputs))
            {
                var output = results.First().AsEnumerable<float>().ToArray();
                int predictedIndex = Array.IndexOf(output, output.Max());
                return classLabels[predictedIndex];
            }
        }

        private void btnChooseImage2_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image files (*.jpg, *.jpeg, *.png)|*.jpg;*.jpeg;*.png";

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                imagePath2 = ofd.FileName;
                pictureBox2.Image = Image.FromFile(imagePath2);
                lblResult2.Text = "Ảnh đã chọn: " + Path.GetFileNameWithoutExtension(imagePath2);
            }
        }
    }
}