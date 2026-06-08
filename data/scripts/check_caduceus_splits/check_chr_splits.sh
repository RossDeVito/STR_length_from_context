# the exact split file Caduceus/HyenaDNA pretrained against
curl -s https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed \
  | awk '{print $1"\t"$4}' | sort -u | sort -k1,1V