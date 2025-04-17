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

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            string modelPath = "my_model.onnx";
            string imagePath = "IMG_20211106_121111 (Custom).jpg";

            // Load và resize ảnh về đúng kích thước 224x224
            Bitmap bitmap = new Bitmap(Image.FromFile(imagePath));
            Bitmap resized = new Bitmap(bitmap, new Size(224, 224));

            // ✅ Lưu ảnh resize để kiểm tra bằng mắt
            resized.Save("resized_debug.jpg");

            // Tạo tensor input [1, 224, 224, 3]
            var inputTensor = new DenseTensor<float>(new[] { 1, 224, 224, 3 });

            for (int y = 0; y < 224; y++)
            {
                for (int x = 0; x < 224; x++)
                {
                    Color pixel = resized.GetPixel(x, y);
                    inputTensor[0, y, x, 0] = pixel.R ;
                    inputTensor[0, y, x, 1] = pixel.G ;
                    inputTensor[0, y, x, 2] = pixel.B ;
                }
            }

            // ✅ In mẫu pixel sau chia 255
            Console.WriteLine("=== 🟦 Sample Input Pixel Values ===");
            for (int i = 0; i < 5; i++)
            {
                Console.WriteLine($"[{i}] R: {inputTensor[0, i, i, 0]:F3} G: {inputTensor[0, i, i, 1]:F3} B: {inputTensor[0, i, i, 2]:F3}");
            }

            // Mean/Std (nếu model cần 3 input)
            var meanTensor = new DenseTensor<float>(new[] { 1, 1, 1, 3 });
            meanTensor[0, 0, 0, 0] = 0.485f;
            meanTensor[0, 0, 0, 1] = 0.456f;
            meanTensor[0, 0, 0, 2] = 0.406f;

            var stdTensor = new DenseTensor<float>(new[] { 1, 1, 1, 3 });
            stdTensor[0, 0, 0, 0] = 0.229f;
            stdTensor[0, 0, 0, 1] = 0.224f;
            stdTensor[0, 0, 0, 2] = 0.225f;

            using (var session = new InferenceSession(modelPath))
            {
                // ✅ Hiển thị thông tin model input/output
                Console.WriteLine("=== 📦 Model Inputs ===");
                foreach (var inputMeta in session.InputMetadata)
                {
                    Console.WriteLine($"📥 {inputMeta.Key}: [{string.Join(", ", inputMeta.Value.Dimensions)}] - {inputMeta.Value.ElementType}");
                }

                Console.WriteLine("=== 📤 Model Outputs ===");
                foreach (var outputMeta in session.OutputMetadata)
                {
                    Console.WriteLine($"📤 {outputMeta.Key}");
                }

                // Bạn có thể thử comment phần mean/std nếu nghi ngờ model không cần
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_layer_1", inputTensor)
                    //NamedOnnxValue.CreateFromTensor("sequential_1/efficientnetb5_1/normalization_1/Sub/y:0", meanTensor),
                    //NamedOnnxValue.CreateFromTensor("sequential_1/efficientnetb5_1/normalization_1/Sqrt/x:0", stdTensor),
                };

                using (var results = session.Run(inputs))
                {
                    foreach (var r in results)
                        Console.WriteLine($"[DEBUG] Output: {r.Name}");
                    var output = results.First().AsEnumerable<float>().ToArray();
                    Console.WriteLine("\n=== 📊 Model Output Scores ===");
                    for (int i = 0; i < classLabels.Length; i++)
                    {
                        Console.WriteLine($"{classLabels[i],-20}: {output[i]:F6}");
                    }

                    // Tìm nhãn có score cao nhất
                    int predictedIndex = Array.IndexOf(output, output.Max());
                    string predictedLabel = classLabels[predictedIndex];

                    // Hiển thị top 5 dự đoán
                    Console.WriteLine("\n=== 🔝 Top 5 Predictions ===");
                    var sorted = output
                        .Select((val, idx) => new { Label = classLabels[idx], Score = val })
                        .OrderByDescending(x => x.Score)
                        .Take(5);

                    foreach (var item in sorted)
                    {
                        Console.WriteLine($"✅ {item.Label}: {item.Score:F4}");
                    }

                    MessageBox.Show($"✅ Dự đoán: {predictedLabel} (Score: {output[predictedIndex]:F4})");
                }
                var p = inputTensor[0, 0, 0, 0];
                Console.WriteLine($"🔎 Pixel [0,0] - R: {inputTensor[0, 0, 0, 0]:F3}, G: {inputTensor[0, 0, 0, 1]:F3}, B: {inputTensor[0, 0, 0, 2]:F3}");

                // Kiểm tra mean/std
                Console.WriteLine($"Mean: {meanTensor[0, 0, 0, 0]} {meanTensor[0, 0, 0, 1]} {meanTensor[0, 0, 0, 2]}");
                Console.WriteLine($"Std : {stdTensor[0, 0, 0, 0]} {stdTensor[0, 0, 0, 1]} {stdTensor[0, 0, 0, 2]}");
                foreach (var input in session.InputMetadata)
                    Console.WriteLine($"🟩 Input: {input.Key} - Shape: {string.Join(",", input.Value.Dimensions)}");

                foreach (var output in session.OutputMetadata)
                    Console.WriteLine($"🟥 Output: {output.Key}");
                
            }
        }
    }
}