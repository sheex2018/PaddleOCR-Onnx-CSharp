namespace PaddleOCRTestOnnx
{
    partial class PaddleOcrOnnxForm
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            buttonClassifierInit = new Button();
            textBoxClassifierFile = new TextBox();
            label1 = new Label();
            label2 = new Label();
            textBoxImageFile = new TextBox();
            buttonImageExplorer = new Button();
            label3 = new Label();
            textBoxDetModel = new TextBox();
            buttonDetModelInfer = new Button();
            buttonDetModelInit = new Button();
            label4 = new Label();
            textBoxRecognitionFile = new TextBox();
            buttonRecognitionInit = new Button();
            textBoxResults = new TextBox();
            label5 = new Label();
            SuspendLayout();
            // 
            // buttonClassifierInit
            // 
            buttonClassifierInit.Location = new Point(344, 96);
            buttonClassifierInit.Name = "buttonClassifierInit";
            buttonClassifierInit.Size = new Size(72, 25);
            buttonClassifierInit.TabIndex = 0;
            buttonClassifierInit.Text = "Init";
            buttonClassifierInit.UseVisualStyleBackColor = true;
            buttonClassifierInit.Click += buttonClassifierInit_Click;
            // 
            // textBoxClassifierFile
            // 
            textBoxClassifierFile.Location = new Point(106, 95);
            textBoxClassifierFile.Name = "textBoxClassifierFile";
            textBoxClassifierFile.Size = new Size(232, 23);
            textBoxClassifierFile.TabIndex = 2;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(40, 100);
            label1.Name = "label1";
            label1.Size = new Size(56, 17);
            label1.TabIndex = 3;
            label1.Text = "方向分类";
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(56, 24);
            label2.Name = "label2";
            label2.Size = new Size(40, 17);
            label2.TabIndex = 6;
            label2.Text = "图  像";
            // 
            // textBoxImageFile
            // 
            textBoxImageFile.Location = new Point(106, 24);
            textBoxImageFile.Name = "textBoxImageFile";
            textBoxImageFile.Size = new Size(232, 23);
            textBoxImageFile.TabIndex = 5;
            // 
            // buttonImageExplorer
            // 
            buttonImageExplorer.Location = new Point(344, 24);
            buttonImageExplorer.Name = "buttonImageExplorer";
            buttonImageExplorer.Size = new Size(72, 25);
            buttonImageExplorer.TabIndex = 4;
            buttonImageExplorer.Text = ">>";
            buttonImageExplorer.UseVisualStyleBackColor = true;
            buttonImageExplorer.Click += buttonImageExplorer_Click;
            // 
            // label3
            // 
            label3.AutoSize = true;
            label3.Location = new Point(40, 59);
            label3.Name = "label3";
            label3.Size = new Size(56, 17);
            label3.TabIndex = 10;
            label3.Text = "区域检测";
            // 
            // textBoxDetModel
            // 
            textBoxDetModel.Location = new Point(106, 56);
            textBoxDetModel.Name = "textBoxDetModel";
            textBoxDetModel.Size = new Size(232, 23);
            textBoxDetModel.TabIndex = 9;
            // 
            // buttonDetModelInfer
            // 
            buttonDetModelInfer.Location = new Point(462, 26);
            buttonDetModelInfer.Name = "buttonDetModelInfer";
            buttonDetModelInfer.Size = new Size(91, 23);
            buttonDetModelInfer.TabIndex = 8;
            buttonDetModelInfer.Text = "Inference";
            buttonDetModelInfer.UseVisualStyleBackColor = true;
            buttonDetModelInfer.Click += buttonDetModelInfer_Click;
            // 
            // buttonDetModelInit
            // 
            buttonDetModelInit.Location = new Point(344, 53);
            buttonDetModelInit.Name = "buttonDetModelInit";
            buttonDetModelInit.Size = new Size(72, 25);
            buttonDetModelInit.TabIndex = 7;
            buttonDetModelInit.Text = "Init";
            buttonDetModelInit.UseVisualStyleBackColor = true;
            buttonDetModelInit.Click += buttonDetModelInit_Click;
            // 
            // label4
            // 
            label4.AutoSize = true;
            label4.Location = new Point(41, 138);
            label4.Name = "label4";
            label4.Size = new Size(56, 17);
            label4.TabIndex = 14;
            label4.Text = "文字识别";
            // 
            // textBoxRecognitionFile
            // 
            textBoxRecognitionFile.Location = new Point(106, 133);
            textBoxRecognitionFile.Name = "textBoxRecognitionFile";
            textBoxRecognitionFile.Size = new Size(232, 23);
            textBoxRecognitionFile.TabIndex = 13;
            // 
            // buttonRecognitionInit
            // 
            buttonRecognitionInit.Location = new Point(344, 133);
            buttonRecognitionInit.Name = "buttonRecognitionInit";
            buttonRecognitionInit.Size = new Size(72, 25);
            buttonRecognitionInit.TabIndex = 11;
            buttonRecognitionInit.Text = "Init";
            buttonRecognitionInit.UseVisualStyleBackColor = true;
            buttonRecognitionInit.Click += buttonRecognitionInit_Click;
            // 
            // textBoxResults
            // 
            textBoxResults.Location = new Point(43, 192);
            textBoxResults.Multiline = true;
            textBoxResults.Name = "textBoxResults";
            textBoxResults.ReadOnly = true;
            textBoxResults.ScrollBars = ScrollBars.Vertical;
            textBoxResults.Size = new Size(375, 234);
            textBoxResults.TabIndex = 15;
            textBoxResults.WordWrap = false;
            // 
            // label5
            // 
            label5.AutoSize = true;
            label5.Location = new Point(41, 172);
            label5.Name = "label5";
            label5.Size = new Size(56, 17);
            label5.TabIndex = 16;
            label5.Text = "识别结果";
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(575, 433);
            Controls.Add(label5);
            Controls.Add(textBoxResults);
            Controls.Add(label4);
            Controls.Add(textBoxRecognitionFile);
            Controls.Add(buttonRecognitionInit);
            Controls.Add(label3);
            Controls.Add(textBoxDetModel);
            Controls.Add(buttonDetModelInfer);
            Controls.Add(buttonDetModelInit);
            Controls.Add(label2);
            Controls.Add(textBoxImageFile);
            Controls.Add(buttonImageExplorer);
            Controls.Add(label1);
            Controls.Add(textBoxClassifierFile);
            Controls.Add(buttonClassifierInit);
            Name = "Form1";
            Text = "Form1";
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button buttonClassifierInit;
        private TextBox textBoxClassifierFile;
        private Label label1;
        private Label label2;
        private TextBox textBoxImageFile;
        private Button buttonImageExplorer;
        private Label label3;
        private TextBox textBoxDetModel;
        private Button buttonDetModelInfer;
        private Button buttonDetModelInit;
        private Label label4;
        private TextBox textBoxRecognitionFile;
        private Button buttonRecognitionInit;
        private TextBox textBoxResults;
        private Label label5;
    }
}