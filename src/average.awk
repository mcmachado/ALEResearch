BEGIN {
    m=int((n+1)/2)
}
{L[NR]=$3; sum+=$3}
NR>=m {d[++i]=$2}
NR>n {sum-=L[NR-n]}
NR>=n{
    a[++k]=sum/n
}
END {
    for (j=1; j<=k; j++)
        print d[j],a[j]
}
