#!/bin/bash
cur_tag=$(curl -sX GET "https://api.github.com/repos/$GITHUB_REPOSITORY/releases/latest"| grep -Po '"tag_name": "\K.*?(?=")')
if [ -z "$cur_tag" ]
then
      cur_tag="v0.1"
else
      cur_tag=$(python3 -c "t=str('$cur_tag');print('v'+'.'.join(t.replace('v','').split('.')[:-1])+'.'+str(int(t.split('.')[-1])+1))")
fi
echo $cur_tag