import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_TRANSLUCIDO_008_material
        objectName: "PLASTICO_TRANSLUCIDO.008"
        baseColor: "#ffccc49f"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Left_Hand_Right_Finger_obj
        objectName: "Nao6_Left_Hand_Right_Finger.obj"
        Model {
            id: nao6_Left_Hand_Right_Finger
            objectName: "Nao6_Left_Hand_Right_Finger"
            source: "meshes/nao6_Left_Hand_Right_Finger_mesh.mesh"
            materials: [
                plastico_TRANSLUCIDO_008_material
            ]
        }
    }

    // Animations:
}
