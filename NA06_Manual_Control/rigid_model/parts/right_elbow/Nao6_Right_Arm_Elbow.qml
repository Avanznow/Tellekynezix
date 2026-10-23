import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: gris_014_material
        objectName: "GRIS.014"
        baseColor: "#ff151515"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Right_Arm_Elbow_obj
        objectName: "Nao6_Right_Arm_Elbow.obj"
        Model {
            id: nao6_Right_Arm_Elbow
            objectName: "Nao6_Right_Arm_Elbow"
            source: "meshes/nao6_Right_Arm_Elbow_mesh.mesh"
            materials: [
                gris_014_material
            ]
        }
    }

    // Animations:
}
