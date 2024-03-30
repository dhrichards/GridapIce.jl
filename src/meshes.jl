function create_mesh(L,n,periodic)
    D = length(L)+1
    if D == 2
        box = CartesianDiscreteModel((0,L[1],0,1),n,isperiodic=(periodic[1],periodic[2]))
        # surface = CartesianDiscreteModel((0,L[1]),(n[1]),isperiodic=(periodic[1],))
    else
        box = CartesianDiscreteModel((0,L[1],0,L[2],0,1),n,isperiodic=periodic)
        # surface = CartesianDiscreteModel((0,L[1],0,L[2]),(n[1],n[2]),isperiodic=(periodic[1],periodic[2]))
    end

    labels = get_face_labeling(box)
    if D == 2
        add_tag_from_tags!(labels,"bottom",[5])
        add_tag_from_tags!(labels,"top",[6])
    else
        add_tag_from_tags!(labels,"bottom",[1,2,3,4,9,10,13,14,21])
        add_tag_from_tags!(labels,"top",[5,6,7,8,11,12,15,16,22])
    end




    return box,labels
end 

function get_labels(box,D)
    labels = get_face_labeling(box)
    if D == 2
        add_tag_from_tags!(labels,"bottom",[5])
        add_tag_from_tags!(labels,"top",[6])
    else
        add_tag_from_tags!(labels,"bottom",[1,2,3,4,9,10,13,14,21])
        add_tag_from_tags!(labels,"top",[5,6,7,8,11,12,15,16,22])
    end
    return labels
end

