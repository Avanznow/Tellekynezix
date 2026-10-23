import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_TRANSLUCIDO_007_material
        objectName: "PLASTICO_TRANSLUCIDO.007"
        baseColor: "#ffccc49f"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Left_Hand_Left_Finger_obj
        objectName: "Nao6_Left_Hand_Left_Finger.obj"
        Model {
            id: nao6_Left_Hand_Left_Finger
            objectName: "Nao6_Left_Hand_Left_Finger"
            source: "meshes/nao6_Left_Hand_Left_Finger_mesh.mesh"
            materials: [
                plastico_TRANSLUCIDO_007_material
            ]
        }
    }

    // Animations:
}
