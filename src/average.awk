BEGIN {
    m=int((n+1)/2)
}
{L[NR]=$2; sum+=$2}
NR>=m {d[++i]=$1}
NR>n {sum-=L[NR-n]}
NR>=n{
    a[++k]=sum/n
}
END {
    for (j=1; j<=k; j++)
        print d[j],a[j]
}
