import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: gris_002_material
        objectName: "GRIS.002"
        baseColor: "#ff151515"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Left_Arm_Elbow_obj
        objectName: "Nao6_Left_Arm_Elbow.obj"
        Model {
            id: nao6_Left_Arm_Elbow
            objectName: "Nao6_Left_Arm_Elbow"
            source: "meshes/nao6_Left_Arm_Elbow_mesh.mesh"
            materials: [
                gris_002_material
            ]
        }
    }

    // Animations:
}
