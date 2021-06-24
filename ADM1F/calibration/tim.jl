import DelimitedFiles
import OrderedCollections

function madsmodelrun_adm1(mddata::AbstractDict)
    # function runs the ADM1F model and return selected values from the last result file as predictions 
    
    function get_last_indicator()
        # finds the last indicator file in the current folder
        flist = readdir()
        last_indicator = ""
        for i = 1:length(flist)
            if length(flist[i]) > 9
                if flist[i][1:9] == "indicator"
                    last_indicator = flist[i]
                end
            end
        end
        return last_indicator
    end
    
    #reads .dat and saves it in mddata
    DelimitedFiles.writedlm("params_tim.dat", Mads.getparamsinit(mddata)) 
    
    #removes all previous .out files
    foreach(rm, filter(endswith(".out"), readdir()))
    
    #runs the ADM1F model with the command line below
    #change the path to the executable below
    command_line=`/Users/elchin/project/ADM1F_WM/build/adm1f -steady -ts_monitor 
             -influent_file influent_tim.dat -params_file params_tim.dat  
             -ts_type beuler -ts_adapt_type basic -snes_rtol 1.e-5 
             -ts_max_snes_failures -1 -Vliq 4981 -t_resx 240` 
    Mads.runcmd(command_line; quiet=true, pipe=true)

    #load the outputs from last indicator 
    #we are interested only in 7 output values out of 67
    #we return those 7 values as predictions 
    last_indicator = get_last_indicator()
    if last_indicator != ""
        o = DelimitedFiles.readdlm(last_indicator)[3:end,:]
        outputs = OrderedCollections.OrderedDict{String, Float64}("o$i"=>o[i] for i=1:67)
        sCOD=0
        for i = 1:7
            sCOD=sCOD+outputs["o$i"]
        end
        Acetate=outputs["o7"]
        Propionate = outputs["o6"]
        Butyrate = outputs["o5"]
        Valerate = outputs["o4"]
        ph = outputs["o27"]
        gaseous = outputs["o41"]*outputs["o44"]*1000
    else
        @error "Output is missing!"
    end
    p=Dict([("o1", sCOD),
            ("o2", Acetate),
            ("o3", Propionate),
            ("o4", Butyrate),
            ("o5", Valerate),
            ("o6", ph),
            ("o7", gaseous)])
    #println("p:",p)
    return p
end
