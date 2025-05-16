#/bin/bash
#

menu(){
	echo "======================================================================"
	echo "注意：请保持该脚本与dockerfile等文件同目录，不要擅自修改目录"
	echo "======================================================================"
	echo ""
	echo "1. 只安装Flask框架，不安装nginx，可通过外部nginx反向代理（强烈推荐）"
	echo "2. Flask + nginx"
	echo "3. 自行下载模型（.pth 文件, 同级目录下的tar压缩包里面）"
	echo ""

	read -p "请选择安装方式（ 1,2 均为docker安装）: " selection
}


flag=0

while [ $flag -ne 1 ];
do
	menu
	case $selection in 
		1) 
			echo "选择1安装中..."
			echo "=========================================================================================================================="
			echo ""
			docker build -t slider_model_flask .
			docker run -itd --name slider_model_flask_ct -p 50000:5000 -v /flask_app/slider_model/images:/flask_app/slider_model/saved_images \
				-v /flask_app/slider_model/json_tmp:/flask_app/slider_model/saved_coco_json slider_model_flask
			echo""
			echo "=========================================================================================================================="

			echo "安装完成，端口为50000，你可以查看映射保存的路径："
			echo "数据标签：/flask_app/slider_model/json_tmp"
			echo "数据图片：/flask_app/slider_model/saved_images"


			flag=1;;
		2)
			echo "选择2安装中... "
			echo "========================================================================================================================"
			echo ""

			touch /var/log/slider_model_flask.log
			touch /var/log/slider_model_flask.error.log
			chmod 644 /var/log/slider_*

			docker compose up -d 

			echo ""
			echo "========================================================================================================================"
			echo "安装完成"
			cat << EOF
注意事项：
1. 你需要自行到当前目录下的etc/ssl/ 中自行添加证书，并重启容器
2. 很多情况下，你需要查询当前网页的ip，你需要使用 docker inspect 容器名 来查询当前容器所属ip
   并将当前目录下 slider_model.conf 中代理ip更改成容器当前ip，使用docker compose restart 来重启容器
   记得同时修改 server_name 为你的域名
3. 当前网页的nginx日志被映射在宿主机的/var/log/目录下，名称为：slider_model_flask
4. 保存的图片以及标签数据在 /flask_app/slider_model/ 下的 saved_images json_tmp中，当然，你也可以在docker-compose.yml中自定义映射路径
5. 还是那句话，不推荐使用这种方式来安装
6. 有任何问题请去github联系我，让你致敬！
EOF

			flag=1 ;;
		3)
			echo "祝你一路顺利孩子..."
			flag=1 ;;
		*)
			echo "请重新选择";;
	esac
done
	
