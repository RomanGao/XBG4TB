## 实验的第一步 将orign_data 转换为1_data
1、对pff.csv数据进行过滤，部分属性提取，再过来得到了pff_subcol.csv
2 、对msf.csv进行过滤，部分属性提取，过滤 得到数据msf_subcol.csv
3、将pff_subcol.DF与msf_col.DF通过id做merge操作 得到pff_msf_labal.csv
4、将pff_msf进行分割成两个数据集，一个是针对基因的数据集pff_msf_gene.csv
  另一部分是基因变异的数据集pff_msf_nu.csv