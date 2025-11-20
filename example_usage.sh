#!/bin/bash
# 示例：如何使用 generate_classification_csv.py 脚本

echo "=========================================="
echo "分类CSV生成脚本使用示例"
echo "=========================================="
echo ""

# 显示帮助信息
echo "1. 查看帮助信息:"
echo "   python generate_classification_csv.py --help"
echo ""

# 显示测试模式
echo "2. 查看测试模式说明:"
echo "   python generate_classification_csv.py --test"
echo ""

# 示例使用命令
echo "3. 实际使用示例:"
echo ""
echo "   假设您有以下目录结构:"
echo "   positive_samples/    # 包含标签为1的CIF文件"
echo "   negative_samples/    # 包含标签为0的CIF文件"
echo ""
echo "   运行命令:"
echo "   python generate_classification_csv.py \\"
echo "       --class1_dir ./positive_samples \\"
echo "       --class0_dir ./negative_samples \\"
echo "       --output my_classification_data.csv"
echo ""

echo "=========================================="
echo "准备您的数据:"
echo "=========================================="
echo ""
echo "请按以下步骤准备您的CIF文件:"
echo ""
echo "1. 创建两个目录用于存放不同类别的CIF文件"
echo "   mkdir class1_cifs  # 标签为1的样本"
echo "   mkdir class0_cifs  # 标签为0的样本"
echo ""
echo "2. 将CIF文件放入对应的目录"
echo "   - 所有标签为1的CIF文件放入 class1_cifs/"
echo "   - 所有标签为0的CIF文件放入 class0_cifs/"
echo ""
echo "3. 运行脚本生成CSV"
echo "   python generate_classification_csv.py \\"
echo "       --class1_dir ./class1_cifs \\"
echo "       --class0_dir ./class0_cifs \\"
echo "       --output classification_data.csv"
echo ""

echo "=========================================="
echo "输出文件说明:"
echo "=========================================="
echo ""
echo "生成的CSV文件将包含以下列:"
echo "  - Id: 唯一标识符"
echo "  - Composition: 化学组成"
echo "  - prop: 分类标签 (1 或 0)"
echo "  - Description: 结构描述文本"
echo "  - File_Name: CIF文件名"
echo ""

# 如果需要，可以在这里添加实际的测试命令
# 但需要确保有测试数据
echo "=========================================="
echo "要查看脚本的详细说明，请参阅:"
echo "  CLASSIFICATION_CSV_README.txt"
echo "=========================================="
