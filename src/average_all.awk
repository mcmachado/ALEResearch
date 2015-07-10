BEGIN{
    m=int((n+1)/2);
}
{L[NR]=$3; sum+=$3;
    d[++i]=$2;
}
NR>n {sum-=L[NR-n];a[++k]=sum/n;}
NR<=n && NR>=m { a[++k]=sum/NR;}
END {
    for (j=1; j<=k; j++)
        print d[j],a[j]
    t=1
    for (j=k+1; j<=NR ; j++){
        sum-=L[NR-n+t];
        t=t+1
        print d[j],sum/(n-t+1)
    }
    
}
