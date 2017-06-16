use strict;
use warnings;

my $coeff = 80;

$coeff = $ARGV[0] if defined $ARGV[0];

my @files = ();

$files[$_] = (2 ** $_) for 0..10;

my %time = ();

my %accel = ();

my %param = ();

my %abs_err = ();
my %rel_err = ();

for my $name(@files){
    my $fh;
    
    if($name == 1){
        open($fh, "<", $name.".out") or last;
    }else{
        open($fh, "<", $name.".out") or next;
    }
    
    while(<$fh>){
        if(/^Time:\s+(.+)$/){
            $time{$name} = $1;
        }elsif(/^Abs. error:\s+(.+)$/){
            $abs_err{$name} = $1;
        }elsif(/^Rel. error:\s+(.+)$/){
            $rel_err{$name} = $1;
        }
        
        if(/^0:\s+(.+):\s+(.+)$/){
            $param{$1} = {} unless exists($param{$1});
            
            $param{$1}->{$name} = $2;
        }
    }
    
    close($fh);
}

if(keys %time){
    return unless defined $time{1};
    
    for(keys %time){
        $accel{$_} = $time{1} / $time{$_};
        $accel{$_} /= 4 if $_ != 1;
    }
    
    printf("%-10s %-15s %-25s %-25s %-25s %-20s %-20s\n", "Nodes", "Cores", "Time", "Speed Up", "Efficiency, %", "Abs. error", "Rel. error");
    
    for(sort {$a <=> $b} keys %time){
        if($_ == 1){
            printf("%-10s %-15s %-25s %-25s %-25s %-20s %-20s\n", $_, $_, $time{$_}, $accel{$_}, ($accel{$_} / $_ * 100), $abs_err{$_}, $rel_err{$_});
        }else{
            printf("%-10s %-15s %-25s %-25s %-25s %-20s %-20s\n", $_, $_ * 4, $time{$_}, $accel{$_}, ($accel{$_} / $_ * 100), $abs_err{$_}, $rel_err{$_});
        }
    }
}else{
    print "No tasks completed\n";
}

print "\n";

if(keys %param){
    for my $i(sort {$a cmp $b} keys %param){
        return unless defined $param{$i}{1};

        my $h = $param{$i};
        
        my $show = 0;
        
        for(keys %{$h}){
            next if $h->{$_} == 0;
            
            if($h->{1} / $h->{$_} < $_ * $coeff / 100){
                $show = 1;
            }
        }
        
        if($show){
            printf("%-10s %-25s %-25s", "np", $i, "%");
            print "\n";
    
            for(sort {$a <=> $b} keys %{$h}){
                printf("%-10s %-25s %-25s\n", $_, $h->{$_}, $h->{1} / $h->{$_});
            }
            
            print "\n";
        }
    }
}
